#
# Copyright 2025 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
kraken.repo
~~~~~~~~~~~

Wrappers around the htrmopo reference implementation implementing
kraken-specific filtering for repository querying operations.
"""
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional, TypeVar, Literal


from htrmopo import get_description as mopo_get_description
from htrmopo import get_listing as mopo_get_listing
from htrmopo.record import v0RepositoryRecord, v1RepositoryRecord


_v0_or_v1_Record = TypeVar('_v0_or_v1_Record', v0RepositoryRecord, v1RepositoryRecord)


def get_description(model_id: str,
                    callback: Callable[..., Any] = lambda: None,
                    version: Optional[Literal['v0', 'v1']] = None,
                    filter_fn: Optional[Callable[[_v0_or_v1_Record], bool]] = lambda x: True) -> _v0_or_v1_Record:
    """
    Filters the output of htrmopo.get_description with a custom function.

    Args:
        model_id: model DOI
        callback: Progress callback
        version:
        filter_fn: Function called to filter the retrieved record.
    """
    desc = mopo_get_description(model_id, callback, version)
    if not filter_fn(desc):
        raise ValueError(f'Record {model_id} exists but is not a valid kraken record')
    return desc


def get_listing(callback: Callable[[int, int], Any] = lambda total, advance: None,
                from_date: Optional[str] = None,
                filter_fn: Optional[Callable[[_v0_or_v1_Record], bool]] = lambda x: True) -> dict[str, dict[str, _v0_or_v1_Record]]:
    """
    Returns a filtered representation of the model repository grouped by
    concept DOI.

    Args:
        callback: Progress callback
        from_data:
        filter_fn: Function called for each record object

    Returns:
        A dictionary mapping group DOIs to one record object per deposit. The
        record of the highest available schema version is retained.
    """
    kwargs = {}
    if from_date is not None:
        kwargs['from'] = from_date
    repository = mopo_get_listing(callback, **kwargs)
    # aggregate models under their concept DOI
    concepts = defaultdict(list)
    for item in repository.values():
        # filter records here
        item = {k: v for k, v in item.items() if filter_fn(v)}
        # both got the same DOI information
        record = item.get('v1', item.get('v0', None))
        if record is not None:
            concepts[record.concept_doi].append(record)

    for k, v in concepts.items():
        concepts[k] = sorted(v, key=lambda x: x.publication_date, reverse=True)

    return concepts
