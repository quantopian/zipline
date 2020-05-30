#
# Copyright 2015 Quantopian, Inc.
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

import pickle


# Label for the serialization version field in the state returned by
# __getstate__.
VERSION_LABEL = '_stateversion_'
CHECKSUM_KEY = '__state_checksum'


def load_context(state_file_path, context, checksum):
    with open(state_file_path, 'rb') as f:
        try:
            loaded_state = pickle.load(f)
        except (pickle.UnpicklingError, IndexError):
            raise ValueError("Corrupt state file: {}".format(state_file_path))
        else:
            if CHECKSUM_KEY not in loaded_state or \
               loaded_state[CHECKSUM_KEY] != checksum:
                raise TypeError("Checksum mismatch during state load. "
                                "The given state file was not created "
                                "for the algorithm in use")
            else:
                del loaded_state[CHECKSUM_KEY]

            for k, v in loaded_state.items():
                setattr(context, k, v)


def store_context(state_file_path, context, checksum, exclude_list):
    state = {}
    fields_to_store = list(set(context.__dict__.keys()) -
                           set(exclude_list))

    for field in fields_to_store:
        state[field] = getattr(context, field)

    state[CHECKSUM_KEY] = checksum

    with open(state_file_path, 'wb') as f:
        # Forcing v2 protocol for compatibility between py2 and py3
        pickle.dump(state, f, protocol=2)
