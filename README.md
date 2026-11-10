# Research notes

Password-protected static site. Every page, table and figure is encrypted at build time
(AES-256-GCM, key from PBKDF2-SHA256 with 600,000 iterations and a per-build random salt)
and decrypted in the browser after the password is entered. The published research notes are held in `data/site.json`.

[Reproduction pipelines](pipelines/) contain source code, input schemas and execution commands. Frozen data and researcher-level results are kept in the private research workspace.
