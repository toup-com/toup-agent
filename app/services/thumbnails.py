"""Small square-ish previews for images in the file library.

Round 21, item 6. A listing that wants to show what a picture IS had exactly
one option: `/library/files/{id}/preview`, which streams the original bytes.
For a screen of twenty generated images that is twenty full-size PNGs — a
2048×2048 render is 3–5 MB — decoded at 40×40 and thrown away. On a phone it
is the listing's whole cost.

So the library grows a derived artefact: one small JPEG per image, produced
once and cached next to nothing (see :func:`cache_path` — it lives in a
dot-directory under the workspace, out of the library scanner's way).

Three properties this is built around:

* **Never fatal.** A thumbnail that cannot be made is a listing that shows a
  generic icon, which is what every listing did before this existed. Every
  entry point returns ``None`` rather than raising.
* **Derived, and it says so.** The cache key folds in the source's size and
  mtime, so replacing an image's bytes under a stable id produces a new
  thumbnail instead of serving the old one for ever.
* **Bounded work.** Pillow's `draft()` lets a JPEG be decoded at 1/2, 1/4 or
  1/8 scale straight out of the file, so a 4000px source is never fully
  decoded to produce a 320px square.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

#: The long edge of a generated thumbnail. 320 covers a 2× phone list row
#: (and a 3× 96pt grid tile) without being worth caching a second size for.
THUMB_MAX_PX = 320

#: JPEG quality. 82 is the knee: below it the ringing shows on flat generated
#: art, above it the file doubles for nothing a thumbnail can show.
THUMB_QUALITY = 82

#: Source files larger than this are not thumbnailed. A 60 MB TIFF decodes to
#: hundreds of MB of pixels, and a listing must never be the thing that OOMs
#: the container.
MAX_SOURCE_BYTES = 40 * 1024 * 1024

#: Refuse absurd pixel dimensions even inside the byte cap — a highly
#: compressible 30 000 × 30 000 PNG is a decompression bomb.
MAX_SOURCE_PIXELS = 80_000_000

#: Where the cache lives. A dot-directory so the library's candidate walk
#: skips it by name, and outside `generated/` so it is not a deliverable.
CACHE_DIR = ".thumbnails"

#: Formats worth thumbnailing. SVG is deliberately absent: it is markup, not
#: pixels, and it is already small enough to send as-is.
THUMBNAILABLE = frozenset({
    "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp",
    "image/bmp", "image/tiff", "image/heic", "image/heif", "image/avif",
})

THUMB_MIME = "image/jpeg"


def can_thumbnail(mime: str, name: str = "") -> bool:
    """Is this worth (and safe to) reduce to a thumbnail?"""
    m = (mime or "").split(";")[0].strip().lower()
    if m in THUMBNAILABLE:
        return True
    if m:
        return False
    ext = (name or "").rsplit(".", 1)[-1].lower() if "." in (name or "") else ""
    return ext in {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff",
                   "heic", "heif", "avif"}


def cache_key(source_path: str) -> Optional[str]:
    """A key that moves when the SOURCE bytes move.

    Size and mtime rather than a content hash: this is called on a listing
    path, and re-reading every image to hash it would cost more than the
    thumbnails save.
    """
    try:
        st = os.stat(source_path)
    except OSError:
        return None
    basis = f"{os.path.realpath(source_path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def cache_path(root: str, key: str) -> str:
    """``<root>/.thumbnails/<ab>/<key>.jpg`` — two levels so one tenant with
    thousands of images does not put thousands of entries in one directory."""
    return os.path.join(root, CACHE_DIR, key[:2], f"{key}.jpg")


def _open_reduced(path: str):
    """A Pillow image opened at the smallest scale that still covers the
    target, with the bomb checks applied. None when it cannot be read."""
    try:
        from PIL import Image, ImageOps
    except Exception:  # noqa: BLE001 - pillow missing is "no thumbnails"
        logger.info("[thumbnails] Pillow is not available")
        return None
    try:
        Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS
    except Exception:  # noqa: BLE001
        pass
    try:
        from pillow_heif import register_heif_opener  # type: ignore
        register_heif_opener()
    except Exception:  # noqa: BLE001 - HEIC support is a bonus, not a rule
        pass
    try:
        img = Image.open(path)
        # JPEG-only fast path: decode at 1/2, 1/4 or 1/8 without ever
        # materialising the full-resolution bitmap.
        try:
            img.draft("RGB", (THUMB_MAX_PX, THUMB_MAX_PX))
        except Exception:  # noqa: BLE001 - not a JPEG, or an old Pillow
            pass
        img = ImageOps.exif_transpose(img)
        return img
    except Exception:  # noqa: BLE001 - a corrupt image is not an error here
        logger.info("[thumbnails] could not open %s", os.path.basename(path))
        return None


def generate(source_path: str, dest_path: str) -> Optional[str]:
    """Write a thumbnail for ``source_path`` at ``dest_path``. Path or None."""
    try:
        st = os.stat(source_path)
    except OSError:
        return None
    if st.st_size > MAX_SOURCE_BYTES:
        logger.info("[thumbnails] %s is %d bytes — too large to thumbnail",
                    os.path.basename(source_path), st.st_size)
        return None

    img = _open_reduced(source_path)
    if img is None:
        return None
    try:
        from PIL import Image
        img.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX), Image.LANCZOS)
        if img.mode in ("RGBA", "LA", "P"):
            # JPEG has no alpha. White rather than black: generated art and
            # screenshots are overwhelmingly light, and a transparent PNG
            # composited onto black reads as a broken image.
            flat = Image.new("RGB", img.size, (255, 255, 255))
            rgba = img.convert("RGBA")
            flat.paste(rgba, mask=rgba.split()[-1])
            img = flat
        elif img.mode != "RGB":
            img = img.convert("RGB")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        tmp = dest_path + ".tmp"
        img.save(tmp, "JPEG", quality=THUMB_QUALITY, optimize=True,
                 progressive=True)
        os.replace(tmp, dest_path)
        return dest_path
    except Exception:  # noqa: BLE001 - see the module docstring
        logger.warning("[thumbnails] could not write a thumbnail for %s",
                       os.path.basename(source_path), exc_info=True)
        try:
            os.unlink(dest_path + ".tmp")
        except OSError:
            pass
        return None
    finally:
        try:
            img.close()
        except Exception:  # noqa: BLE001
            pass


def ensure(root: str, source_path: str) -> Optional[Tuple[str, str]]:
    """``(thumbnail path, cache key)`` for one image, making it if needed."""
    key = cache_key(source_path)
    if not key:
        return None
    dest = cache_path(root, key)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest, key
    made = generate(source_path, dest)
    return (made, key) if made else None
