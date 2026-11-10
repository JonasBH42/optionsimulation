# Dependency license and source review

This is a private project handoff, not a public release or a blanket legal
compliance certificate. No license has been assigned to the user's application
code. Third-party licenses continue to apply to their own components; a private
handoff is not treated as an exemption from those terms.

The portable folder must accompany this review, `THIRD_PARTY_NOTICES.txt`,
`dependency-notices.json`, `licenses/`, and `third_party_sources/` including its
hash manifest. The full project source and build settings are supplied separately
in `D:\optionsimulation`; the market archives are not redistribution materials.

## Components inspected

| Component in this build | Evidence and treatment |
| --- | --- |
| PySide6 / Shiboken 6.12.0 | Installed wheel metadata declares LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only. Its packaged license directory contains only `LicenseRef-Qt-Commercial.txt`; that file does not establish a commercial entitlement. The official PySide source archive, full LGPLv3/GPLv3 texts, and upstream license/exception files accompany this handoff. |
| Qt Base, SVG and Image Formats 6.12.0 | Matching official source archives are supplied. Qt Base covers the observed Core, Gui, Widgets, Network, OpenGL, OpenGLWidgets and Test DLLs. The source archives retain component and bundled-third-party notices. Review individual module/file terms; not every Qt module has the same license choices. |
| Qt PDF 6.140.0 | `qt6pdf.dll` reports 6.140.0.0 and is pulled in by the PDF image-format plugin. Its source is the separate Qt WebEngine/PDF 6.140.0 package from Qt's official SDK repository for Qt 6.12.0, not a nonexistent Qt 6.14 archive. The full archive includes Chromium/PDFium sources and their own notices. |
| CPython 3.12.16 | The actual interpreter's `LICENSE.txt` is collected, including its Windows Microsoft Distributable Code notice. Python's license is distinct from the application and Qt licenses. |
| OpenSSL 3.5.9 | The observed `libssl-3-x64.dll` and `libcrypto-3-x64.dll` report 3.5.9. The official OpenSSL 3.5.9 Apache-2.0 license text is included. Older OpenSSL/SSLeay notices that occur inside another wheel's notices do not replace this component's license. |
| libffi | `libffi-8.dll` is present but exposes no reliable embedded package version. An official libffi 3.4.8 MIT-style license copy is included and labeled as a reference, not proof of the binary's exact source revision. Exact vendor provenance remains an item to confirm before onward redistribution. |
| MSVC runtime | Observed DLLs include MSVCP 14.40.33810.0 and VCRUNTIME 14.44.35211.0. Preserve their vendor notices and the Python Windows redistribution notice. These are Microsoft redistributable components, not covered by Qt's LGPL; the collector does not independently certify a Microsoft redistribution grant. |
| NumPy, SciPy and native numeric libraries | Installed license files and the wheels' bundled OpenBLAS/GCC-runtime and other third-party notices are retained in the collected notices. Consult the exact wheel records rather than applying only the top-level NumPy/SciPy license. |
| PyArrow, DuckDB, pyqtgraph and remaining installed packages | The wheel inventory and extracted license/NOTICE files are in `dependency-notices.json` and `THIRD_PARTY_NOTICES.txt`. Build/test packages are inventoried too; this does not imply all of them are in the portable runtime. |
| Nuitka 4.1.1 | The compiler reports AGPLv3; its collected `LICENSE-RUNTIME.txt` supplies the separate Nuitka Runtime Library Exception. Preserve that exception and its scope rather than assuming that compiling the application assigns it the compiler's license. |

Source archive origin URLs, byte counts and SHA-256 hashes are in
`third_party_sources/manifest.json`. Qt archive hashes are compared with the
official HTTPS mirror-list hashes. Download redirects are recorded where used.
The project has not patched the supplied Qt/PySide libraries. Upstream wheel
build flags and any vendor-specific patches have not been independently
reconstructed, and no byte-for-byte rebuild claim is made. Keep the archives,
their internal notices, and this qualification together.

## Replacing or rebuilding the dynamic libraries

1. Close the application and its worker processes. Make a working copy of the
   entire standalone folder inside `D:\optionsimulation`, for example beneath
   `runtime\library-replacement-test`; retain the original folder.
2. Extract the supplied source archives beneath a project-local rebuild
   directory. Use the upstream build instructions contained in each archive and
   a compatible existing x64 Windows compiler/toolchain. Keep all build outputs
   and temporary files under `D:\optionsimulation`.
3. Build a compatible set of Qt 6.12 libraries/plugins and, if needed, Qt
   WebEngine/PDF 6.140 and PySide/Shiboken 6.12 for Python 3.12. Replace the
   corresponding DLLs/plugins in the copied folder while preserving filenames
   and directory layout. Replace related modules consistently; a different ABI
   may require rebuilding the bindings and application from the supplied source.
4. Run the copied `QQQSimulator.exe doctor` and `QQQSimulator.exe --demo`. The
   application does not require a signature key or vendor approval to load a
   compatible replacement DLL. Model-pack integrity checks are separate from
   this library-replacement process. Use a new test session after changing the
   implementation; existing sessions retain their original implementation
   identity and can still be exported read-only.
5. The project source, lockfile, `packaging/main.py`, generated deployment
   settings, and `packaging/build.ps1` provide the application rebuild path.
   Replacement/rebuild compatibility has not been independently demonstrated for
   arbitrary modified Qt libraries. No term in this review restricts modification
   or reverse engineering permitted by a component's license for debugging such
   changes.

## Scope of the review

Qt's published guidance calls for the relevant license notices, corresponding
library source or a qualifying source offer, and the recipient's ability to
replace/relink the library. This handoff supplies actual source archives rather
than only upstream links. A later distributor must review the final binary
inventory, exact source correspondence, third-party and Microsoft terms, and any
changes to packaging or application licensing; these collected files alone do
not establish every distribution condition.

Official references used for this review:

- [Qt LGPL/GPL obligations](https://www.qt.io/development/open-source-lgpl-obligations)
- [Qt open-source licensing FAQ](https://www.qt.io/faq/qt-open-source-licensing)
- [Separate Qt WebEngine/PDF versioning](https://www.qt.io/blog/qt-webengine-will-release-separate)
- [GNU LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.html)
- [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.html)
