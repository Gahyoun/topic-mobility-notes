# Research notes

Password-protected static site. Every page, table and figure is encrypted at build time
(AES-256-GCM, key from PBKDF2-SHA256 with 600,000 iterations and a per-build random salt)
and decrypted in the browser after the password is entered. This repository holds only the
unlock shell and the ciphertext (`data/site.json`); no plaintext content.
