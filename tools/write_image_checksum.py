# -*- coding: utf-8 -*-
"""Write the image's sha256 sidecar with the **submitted** file name.

The build tool writes the sidecar next to the archive it just produced,
and that archive carries the version in its name.  The submission
package ships the archive under the plain name the platform expects
(``zhixun-engine-linux-amd64.tar``), so the sidecar has to name **that**
file or a reviewer running ``sha256sum -c`` gets a "file not found".
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "提交材料_20261125" / "02_Harness系统镜像"
ARCHIVE = PACKAGE / "zhixun-engine-linux-amd64.tar"
SIDECAR = PACKAGE / "zhixun-engine-linux-amd64.tar.sha256"


def main():
    if not ARCHIVE.is_file():
        print("missing %s" % ARCHIVE)
        return 1
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    hexdigest = digest.hexdigest()
    SIDECAR.write_text("%s  %s\n" % (hexdigest, ARCHIVE.name), encoding="utf-8")
    print("wrote %s" % SIDECAR)
    print("  %s  %s (%d bytes)"
          % (hexdigest, ARCHIVE.name, ARCHIVE.stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
