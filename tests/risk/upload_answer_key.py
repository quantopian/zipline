#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility script for maintainer use to upload current version of the answer key
spreadsheet to S3.
"""
import hashlib

import boto

from . import answer_key

BUCKET_NAME = 'zipline-test-data'


def main():
    with open(answer_key.ANSWER_KEY_PATH, 'r') as f:
        md5 = hashlib.md5()
        while True:
            buf = f.read(1024)
            if not buf:
                break
            md5.update(buf)
    local_hash = md5.hexdigest()

    s3_conn = boto.connect_s3()

    bucket = s3_conn.get_bucket(BUCKET_NAME)
    key = boto.s3.key.Key(bucket)

    key.key = "risk/{local_hash}/risk-answer-key.xlsx".format(
        local_hash=local_hash)
    key.set_contents_from_filename(answer_key.ANSWER_KEY_PATH)
    key.set_acl('public-read')

    download_link = "http://s3.amazonaws.com/{bucket_name}/{key}".format(
        bucket_name=BUCKET_NAME,
        key=key.key)

    print("Uploaded to key: {key}".format(key=key.key))
    print("Download link: {download_link}".format(download_link=download_link))

    # Now update checksum file with the recently added answer key.
    # checksum file update will be then need to be commited via git.
    with open(answer_key.ANSWER_KEY_CHECKSUMS_PATH, 'a') as checksum_file:
        checksum_file.write(local_hash)
        checksum_file.write("\n")

if __name__ == "__main__":
    main()
