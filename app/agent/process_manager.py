"""
Process Manager — Background exec session management.
Start, stop, status, list background processes (long-running commands).
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ManagedProcess:
    """A managed background process."""
    process_id: str
    command: str
    cwd: str = "/tmp"
    state: ProcessState = ProcessState.PENDING
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    stdout_buffer: List[str] = field(default_factory=list)
    stderr_buffer: List[str] = field(default_factory=list)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    max_output_lines: int = 1000
    user_id: str = ""
    session_id: str = ""
    _process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)

    def uptime(self) -> float:
        if not self.started_at:
            return 0
        end = self.ended_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict:
        return {
            "process_id": self.process_id,
            "command": self.command,
            "cwd": self.cwd,
            "state": self.state.value,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "stdout_lines": len(self.stdout_buffer),
            "stderr_lines": len(self.stderr_buffer),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "uptime": self.uptime(),
            "user_id": self.user_id,
            "session_id": self.session_id,
        }


class ProcessManager:
    """Manage background processes for the agent."""

    def __init__(self, max_processes: int = 20):
        self._processes: Dict[str, ManagedProcess] = {}
        self._max_processes = max_processes

    async def start(
        self,
        command: str,
        cwd: str = "/tmp",
        user_id: str = "",
        session_id: str = "",
        env: Optional[Dict[str, str]] = None,
    ) -> ManagedProcess:
        """Start a background process."""
        if len([p for p in self._processes.values() if p.state == ProcessState.RUNNING]) >= self._max_processes:
            raise RuntimeError(f"Max concurrent processes ({self._max_processes}) reached")

        proc = ManagedProcess(
            process_id=str(uuid.uuid4())[:12],
            command=command,
            cwd=cwd,
            user_id=user_id,
            session_id=session_id,
        )
        self._processes[proc.process_id] = proc

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            proc._process = process
            proc.pid = process.pid
            proc.state = ProcessState.RUNNING
            proc.started_at = time.time()

            # Start output readers
            asyncio.create_task(self._read_stream(proc, process.stdout, proc.stdout_buffer))
            asyncio.create_task(self._read_stream(proc, process.stderr, proc.stderr_buffer))
            asyncio.create_task(self._wait_process(proc, process))

            logger.info(f"Started process {proc.process_id} (pid={proc.pid}): {command}")
        except Exception as e:
            proc.state = ProcessState.FAILED
            proc.ended_at = time.time()
            logger.error(f"Failed to start process: {e}")
            raise

        return proc

    async def _read_stream(self, proc: ManagedProcess, stream, buffer: List[str]):
        """Read from a process stream into buffer."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                buffer.append(text)
                if len(buffer) > proc.max_output_lines:
                    buffer.pop(0)
        except Exception:
            pass

    async def _wait_process(self, proc: ManagedProcess, process: asyncio.subprocess.Process):
        """Wait for a process to complete."""
        try:
            exit_code = await process.wait()
            proc.exit_code = exit_code
            proc.state = ProcessState.COMPLETED if exit_code == 0 else ProcessState.FAILED
            proc.ended_at = time.time()
        except Exception:
            proc.state = ProcessState.FAILED
            proc.ended_at = time.time()

    async def stop(self, process_id: str, force: bool = False) -> bool:
        """Stop a running process."""
        proc = self._processes.get(process_id)
        if not proc or not proc._process:
            return False
        if proc.state != ProcessState.RUNNING:
            return False
        try:
            if force:
                proc._process.kill()
            else:
                proc._process.terminate()
            proc.state = ProcessState.STOPPED
            proc.ended_at = time.time()
            return True
        except Exception as e:
            logger.error(f"Failed to stop process {process_id}: {e}")
            return False

    def status(self, process_id: str) -> Optional[dict]:
        """Get process status."""
        proc = self._processes.get(process_id)
        return proc.to_dict() if proc else None

    def output(self, process_id: str, lines: int = 50, stream: str = "stdout") -> Optional[List[str]]:
        """Get recent output from a process."""
        proc = self._processes.get(process_id)
        if not proc:
            return None
        buf = proc.stdout_buffer if stream == "stdout" else proc.stderr_buffer
        return buf[-lines:]

    def list_processes(
        self, state: Optional[ProcessState] = None, session_id: Optional[str] = None
    ) -> List[dict]:
        """List all managed processes."""
        procs = list(self._processes.values())
        if state:
            procs = [p for p in procs if p.state == state]
        if session_id:
            procs = [p for p in procs if p.session_id == session_id]
        return [p.to_dict() for p in sorted(procs, key=lambda p: p.started_at or 0, reverse=True)]

    async def stop_all(self, session_id: Optional[str] = None) -> int:
        """Stop all running processes, optionally filtered by session."""
        count = 0
        for proc in list(self._processes.values()):
            if proc.state == ProcessState.RUNNING:
                if session_id and proc.session_id != session_id:
                    continue
                await self.stop(proc.process_id)
                count += 1
        return count

    def cleanup(self, max_age: float = 3600) -> int:
        """Remove old completed/failed processes."""
        now = time.time()
        removed = 0
        for pid in list(self._processes.keys()):
            proc = self._processes[pid]
            if proc.state in (ProcessState.COMPLETED, ProcessState.FAILED, ProcessState.STOPPED):
                if proc.ended_at and (now - proc.ended_at) > max_age:
                    del self._processes[pid]
                    removed += 1
        return removed


_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    """Get the global process manager singleton."""
    global _manager
    if _manager is None:
        _manager = ProcessManager()
    return _manager
