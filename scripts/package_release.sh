#!/usr/bin/env bash
# Build parakeet.cpp in distribution mode and stage a release tarball.
#
# Usage:
#   scripts/package_release.sh --version v0.1.0
#   scripts/package_release.sh --version v0.1.0 --build-dir build-release
#
# Output: dist/parakeet-<version>-<os>-<arch>.tar.gz
#
# Used by .github/workflows/release.yml. Designed to be runnable locally so
# rpath/packaging changes can be validated before pushing a tag.

set -euo pipefail

VERSION=""
BUILD_DIR="build-release"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$VERSION" ]]; then
    echo "error: --version is required (e.g. v0.1.0)" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ─── Platform slug ──────────────────────────────────────────────────────────
case "$(uname -s)" in
    Darwin)  OS="macos" ;;
    Linux)   OS="linux" ;;
    *) echo "error: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
    arm64|aarch64) ARCH="arm64" ;;
    x86_64)        ARCH="x86_64" ;;
    *) echo "error: unsupported arch $(uname -m)" >&2; exit 1 ;;
esac

SLUG="parakeet-${VERSION}-${OS}-${ARCH}"
STAGE_DIR="dist/${SLUG}"
TARBALL="dist/${SLUG}.tar.gz"

echo "==> Building ${SLUG}"

# ─── Configure + build ──────────────────────────────────────────────────────
# Dist mode statically links libomp/libgomp into libaxiom and OpenBLAS on Linux,
# leaving only system frameworks (Accelerate, Metal) as runtime deps on macOS.
cmake -S . -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAXIOM_DIST_BUILD=ON \
    -DPARAKEET_BUILD_CLI=ON \
    -DPARAKEET_BUILD_TESTS=OFF \
    -DPARAKEET_BUILD_EXAMPLES=ON \
    -DPARAKEET_BUILD_SERVER_EXAMPLE=ON

cmake --build "$BUILD_DIR" -j

# ─── Stage layout ───────────────────────────────────────────────────────────
echo "==> Staging ${STAGE_DIR}"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"/{bin,lib,include}

cp "$BUILD_DIR/parakeet"                       "$STAGE_DIR/bin/"
cp "$BUILD_DIR/examples/server/example-server" "$STAGE_DIR/bin/"

# Copy libaxiom + symlinks (-a preserves symlinks, perms, timestamps).
if [[ "$OS" == "macos" ]]; then
    cp -a "$BUILD_DIR/third_party/axiom/"libaxiom.*.dylib "$STAGE_DIR/lib/"
    cp -a "$BUILD_DIR/third_party/axiom/"libaxiom.dylib   "$STAGE_DIR/lib/"
else
    cp -a "$BUILD_DIR/third_party/axiom/"libaxiom.so*     "$STAGE_DIR/lib/"
fi

cp -a include/parakeet "$STAGE_DIR/include/"
cp LICENSE README.md   "$STAGE_DIR/"

# ─── Fix rpaths ─────────────────────────────────────────────────────────────
# Make binaries find bundled libaxiom relative to their own location, so the
# tarball is fully relocatable.
echo "==> Fixing rpaths"

fix_rpath_macos() {
    local bin="$1"
    # Strip any build-dir rpaths that snuck in (cmake adds them by default).
    local rpaths
    rpaths=$(otool -l "$bin" | awk '/LC_RPATH/{f=1;next} f && /path /{print $2; f=0}')
    while IFS= read -r rp; do
        [[ -z "$rp" ]] && continue
        install_name_tool -delete_rpath "$rp" "$bin" 2>/dev/null || true
    done <<< "$rpaths"
    install_name_tool -add_rpath "@loader_path/../lib" "$bin"
}

fix_rpath_linux() {
    local bin="$1"
    patchelf --set-rpath '$ORIGIN/../lib' "$bin"
}

for bin in "$STAGE_DIR/bin/"*; do
    if [[ "$OS" == "macos" ]]; then
        fix_rpath_macos "$bin"
    else
        fix_rpath_linux "$bin"
    fi
done

# ─── Verify no leaked host paths ────────────────────────────────────────────
echo "==> Verifying"
LEAK=0
for bin in "$STAGE_DIR/bin/"*; do
    if [[ "$OS" == "macos" ]]; then
        if otool -L "$bin" | grep -E "/opt/homebrew|/usr/local/Cellar|$REPO_ROOT" >&2; then
            echo "error: $bin references a host path above" >&2
            LEAK=1
        fi
    else
        if ldd "$bin" 2>/dev/null | grep -E "$REPO_ROOT|/home/" >&2; then
            echo "error: $bin references a host path above" >&2
            LEAK=1
        fi
    fi
done
if [[ "$LEAK" -ne 0 ]]; then
    exit 1
fi

# ─── Tarball ────────────────────────────────────────────────────────────────
echo "==> Creating ${TARBALL}"
tar -czf "$TARBALL" -C dist "$SLUG"

if command -v shasum >/dev/null; then
    shasum -a 256 "$TARBALL"
else
    sha256sum "$TARBALL"
fi

echo "==> Done: ${TARBALL}"
