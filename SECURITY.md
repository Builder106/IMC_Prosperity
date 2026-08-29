# Security notes

## ChromaDB dependency exception

The application uses ChromaDB 1.5.9 through `langchain-chroma` for three local
vector stores. GitHub currently reports four open ChromaDB advisories, with no
patched version available:

- [CVE-2026-45829](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c)
- [CVE-2026-45830](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr)
- [CVE-2026-45831](https://github.com/advisories/GHSA-xph7-9rjv-w5fr)
- [CVE-2026-45833](https://github.com/advisories/GHSA-36p7-vc44-83pf)

The audit tool reports CVE-2026-45829 under the identifier `PYSEC-2026-311`.

The CI dependency audit temporarily ignores only these findings. The exception
assumes that Chroma remains embedded in the application. The application does
not start a Chroma HTTP server, expose Chroma REST endpoints, accept
user-controlled collection or model configuration, or enable Hugging Face
remote code execution.

This exception does not mean that ChromaDB is unused or safe. Remove it when a
patched release is available, or when the application no longer needs Chroma.
Reassess it before introducing a networked Chroma service or loading
user-controlled embedding models.
