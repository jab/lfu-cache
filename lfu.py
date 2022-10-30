#!/usr/bin/env python3

"""
Problem
=======

Implement a least-frequently-used (LFU) cache with capacity *maxsize*.
Inserting a new (key, value) item associates it with frequency 0.
Retrieving the value of a cached key increments the associated frequency by 1.
If capacity has been reached, to make room for the new item,
the item with lowest frequency will first be evicted, and its frequency will be forgotten.
If there are multiple items tied for lowest frequency, the least-recently retrieved is evicted.
Attempting to re-insert an existing (key, value) item is a no-op.
Associating an existing key with a new value should reset the associated frequency to 0.

NB: https://en.wikipedia.org/wiki/Least_frequently_used#Problems


Solution
========

The implementation below has O(1) time complexity for insertion, eviction, and lookup,
and takes O(maxsize) space.

A doubly-linked list is used for frequency buckets, e.g.

  [0]-[1]-[4]-[7]

Each frequency bucket also stores a doubly-linked list of key-value nodes with that frequency:

       frequencies
       ^^^^^^^^^^^
     [0]-[1]-[4]-[7]
      |   |   |   |
 k:  <A> <D> <F> <G>
 e:   |   |       |
 y:  <B> <E>     <H>
 s:   |
     <C>

In this example, A-B-C is the list of all keys with frequency 0,
D-E is the list of all keys with frequency 1, and so forth.
(Values associated with keys are omitted for brevity.)

Each key-value list node ("kvnode") stores a reference back to its frequency list node ("freqnode"),
so e.g. the nodes for keys D and E each have a reference to the node for frequency 1.

Finally, a backing dict associates keys with kvnodes, so that a kvnode can be moved
to the next-highest-frequency list as a result of a lookup in constant time.
Since the least-frequency-used key is always the first item in the frequency list,
insertion and eviction are also constant-time.
"""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import typing as t


KT = t.TypeVar("KT")  # key type
VT = t.TypeVar("VT")  # value type
DT = t.TypeVar("DT")  # node data type


@dataclass
class KVNodeData(t.Generic[KT, VT]):
    key: KT
    val: VT
    freq_node: FreqNode


@dataclass
class FreqNodeData:
    freq: int        # frequency
    kvlhead: KVNode  # head of kvnode list


class AnyNodeData(SimpleNamespace):
    pass


class Node(t.Generic[DT]):
    """Generic doubly-linked list node.

    >>> (head := Node())
    ()
    >>> head.unlinked()
    True
    >>> head.nxt is head.prv is head
    True
    >>> head.insert(Node(x=1))
    >>> head.unlinked()
    False
    >>> head
    ()(x=1)
    >>> head.nxt.data.x
    1
    >>> head.insert(Node(x=2))
    >>> head
    ()(x=2)(x=1)
    """

    data_cls: t.Type[DT] = t.cast(t.Type[DT], AnyNodeData)
    data: DT

    __slots__ = ("prv", "nxt", "data")

    def __init__(self, **data):
        """Create a new unlinked Node."""
        self.prv = self.nxt = self
        self.data = self.data_cls(**data)

    def unlinked(self) -> bool:
        return self.nxt is self and self.prv is self

    def unlink(self) -> None:
        self.prv.nxt = self.nxt
        self.nxt.prv = self.prv
        self.nxt = self.prv = self

    def __repr__(self) -> str:
        s = ""
        cur = start = self
        while not s or (cur := cur.nxt) is not start:
            s += str(cur.data).removeprefix(self.data_cls.__name__)
        return s

    def insert(self, new_nxt: t.Self) -> None:
        """Insert *new_nxt* in between this node and its current next node."""
        old_nxt = self.nxt
        self.nxt = new_nxt
        new_nxt.prv = self
        new_nxt.nxt = old_nxt
        old_nxt.prv = new_nxt


class KVNode(Node):
    data_cls = KVNodeData

    @property
    def freq(self):
        return self.data.freq_node.data.freq


class FreqNode(Node):
    data_cls = FreqNodeData

    def __init__(self, freq):
        kvlhead = KVNode(freq_node=self, key=..., val=...)
        super().__init__(freq=freq, kvlhead=kvlhead)

    def is_empty(self) -> bool:
        return self.data.kvlhead.unlinked()


class LFUCache(t.Generic[KT, VT]):
    """Create an LFUCache with the given *maxsize*.

    >>> (cache := LFUCache(4))
    LFUCache({})
    >>> cache.put("A", "a")
    >>> cache.put("B", "b")
    >>> cache.get("B")
    'b'
    >>> cache
    LFUCache({A: a [freq=0], B: b [freq=1]})
    """

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._node_by_key = {}
        self._freq0_node = FreqNode(freq=0)

    def get(self, key: KT) -> VT:
        """Look up value associated with *key*, and increment the associated frequency.

        Raise KeyError if *key* is not in cache.
        """
        kvnode = self._node_by_key[key]  # raise KeyError if no node with this key
        target_freq_node = self._get_or_create_inc_freq_node(kvnode)
        self._move_to_target_freq(kvnode, target_freq_node)
        return kvnode.data.val

    def _get_or_create_inc_freq_node(self, kvnode: KVNode) -> FreqNode:
        """Given kvnode with frequency f, get or create the freq node for frequency f + 1."""
        cur_freq_node = kvnode.data.freq_node
        target_freq = cur_freq_node.data.freq + 1
        target_freq_node = cur_freq_node.nxt
        if target_freq_node.data.freq != target_freq:
            # need to create the target frequency node
            target_freq_node = FreqNode(freq=target_freq)
            cur_freq_node.insert(target_freq_node)
        return target_freq_node

    def _move_to_target_freq(self, node: KVNode, target_freq_node: FreqNode) -> None:
        self._unlink(node)
        node.data.freq_node = target_freq_node
        target_freq_node.data.kvlhead.insert(node)

    def put(self, key: KT, val: VT, _missing=object()) -> None:
        """Insert (key, val) into cache, evicting the LFU item to make room if necessary."""
        node = self._node_by_key.get(key, _missing)
        if node is _missing:
            node = KVNode(freq_node=self._freq0_node, key=key, val=val)
            self._node_by_key[key] = node
        else:
            assert isinstance(node, KVNode)
            if node.data.val == val:
                return  # (key, val) already inserted -> no-op
            self._unlink(node)
            node.data.val = val
            node.data.freq_node = self._freq0_node
        while len(self) > self.maxsize:  # O(1) (at most 1 iteration) unless maxsize was decreased after init
            self._evict()
        self._freq0_node.data.kvlhead.insert(node)

    def _unlink(self, kvnode: KVNode) -> None:
        """Remove kvnode and prune its frequency bucket if no longer needed."""
        kvnode.unlink()
        freq_node = kvnode.data.freq_node  # Prune associated freq_node if empty...
        # ...but not if it's the freq0 node, since we need the freq0 node on every put:
        if freq_node.is_empty() and freq_node is not self._freq0_node:
            freq_node.unlink()

    def _evict(self) -> None:
        assert self
        if self._freq0_node.is_empty():      # freq0 node may be empty. If so...
            lfu_node = self._freq0_node.nxt  # ...lfu_node must be its nxt, since...
            assert not lfu_node.is_empty()   # ...no other freq nodes may be empty.
        else:
            lfu_node = self._freq0_node
        evict_node = lfu_node.data.kvlhead.prv  # Choose the oldest key in the lfu bucket.
        assert not evict_node.unlinked()
        self._unlink(evict_node)
        del self._node_by_key[evict_node.data.key]

    def __len__(self) -> int:
        return len(self._node_by_key)

    def __contains__(self, key: KT) -> bool:
        return key in self._node_by_key

    def __iter__(self) -> t.Iterator[KT]:
        yield from self._node_by_key

    def freq(self, key: KT) -> int:
        return self._node_by_key[key].freq

    def to_mapping(self) -> t.Mapping[KT, VT]:
        return {k: n.data.val for k, n in self._node_by_key.items()}

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}({{{
            ", ".join(f"{k}: {n.data.val} [freq={self.freq(k)}]"
                      for (k, n) in self._node_by_key.items())
        }}})"""


if __name__ == "__main__":
    import doctest
    doctest.testmod()
