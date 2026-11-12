"""Merge encrypted KOAD release exclusions into private work-level adjudications."""
import argparse
import base64
from contextlib import closing
from getpass import getpass
import gzip
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from filter_namesakes import read_only, work_id


def decode(path, password):
    wrapper = json.loads(path.read_text())
    if wrapper['kdf'] != 'PBKDF2-SHA256' or wrapper['compression'] != 'gzip':
        raise ValueError('Unsupported release decision format.')

    def binary(key):
        return base64.b64decode(wrapper[key], validate=True)

    key = PBKDF2HMAC(SHA256(), 32, binary('salt'), int(wrapper['iterations'])).derive(
        password.encode()
    )
    clear = AESGCM(key).decrypt(binary('nonce'), binary('ciphertext'), binary('aad'))
    return json.loads(gzip.decompress(clear))


def public_id(salt, uid):
    digest = hmac.new(salt, uid.encode(), hashlib.sha256).digest()
    return 'P-' + base64.b32encode(digest).decode()[:10]


def merge(uids, salt, existing, conflict, corroboration):
    identity = {public_id(salt, uid): uid for uid in uids}
    if len(identity) != len(uids):
        raise ValueError('Anonymous roster identifier collision.')
    proposals = {
        'career_conflict': [(spec['professor_uid'], work['work_id'])
                            for spec in conflict for work in spec['rejected_works']],
        'require_corroboration': [(pid, work) for pid, works in corroboration.items()
                                  for work in works],
    }
    additions = []
    counts = {}
    for rule, pairs in proposals.items():
        linked = {(identity[pid], work_id(work)) for pid, work in pairs if pid in identity}
        counts[rule] = {'release_records': len(pairs), 'matched_roster_pairs': len(linked),
                        'records_outside_roster': sum(pid not in identity for pid, _ in pairs)}
        additions.extend({'professor_uid': uid, 'work_id': work, 'decision': 'drop',
                          'rule': rule, 'evidence': f'Archived dashboard release decision: {rule}'}
                         for uid, work in sorted(linked))
    return existing + additions, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'salt', 'existing', 'conflict', 'audit', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    password = getpass('Dashboard release password: ')
    conflict = decode(args.conflict, password)['specs']
    corroboration = decode(args.audit, password)['corroboration']
    with closing(sqlite3.connect(read_only(args.cohort), uri=True)) as db:
        uids = {uid for uid, in db.execute('SELECT professor_uid FROM roster_coverage')}
    records, counts = merge(uids, args.salt.read_bytes(), json.loads(args.existing.read_text()),
                            conflict, corroboration)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(records, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
