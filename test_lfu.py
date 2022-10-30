from hypothesis import assume, given, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule
from hypothesis.strategies import integers, data, sampled_from

from lfu import LFUCache


@settings(max_examples=200, stateful_step_count=100)
class LFUCacheStateMachine(RuleBasedStateMachine):
    @initialize(maxsize=integers(min_value=1, max_value=10))
    def init_lfu(self, maxsize):
        self.maxsize = maxsize
        self.lfu = LFUCache(maxsize=maxsize)
        self.expect_items = {}
        self.expect_freq_by_key = {}

    @invariant()
    def maxsize_not_exceeded(self):
        assert len(self.lfu) <= self.maxsize

    @invariant()
    def items_as_expected(self):
        assert self.lfu.to_mapping() == self.expect_items

    @invariant()
    def freqs_as_expected(self):
        assert {k: self.lfu.freq(k) for k in self.expect_freq_by_key} == self.expect_freq_by_key

    @rule(key=integers(), val=integers())
    def put(self, key, val):
        oldval = self.expect_items.get(key)
        if oldval is not None:
            return self._put_existing(key, val)
        len_before = len(self.lfu)
        at_max = len_before == self.maxsize
        evictable_keys = ()
        if at_max:
            lowest_freq = min(self.expect_freq_by_key.values())
            evictable_keys = {k for k, v in self.expect_freq_by_key.items() if v == lowest_freq}
            assert all(k in self.lfu for k in evictable_keys)
        self.lfu.put(key, val)
        self.expect_items[key] = val
        self.expect_freq_by_key[key] = 0
        assert self.lfu.freq(key) == 0
        if not at_max:
            assert len(self.lfu) == 1 + len_before
            return
        # at max
        evicted_keys = [k for k in evictable_keys if k not in self.lfu]
        assert len(evicted_keys) == 1
        evicted_key, = evicted_keys
        del self.expect_freq_by_key[evicted_key]
        del self.expect_items[evicted_key]
        assert len(self.lfu) == len_before

    def _put_existing(self, key, newval):
        freq_before = self.lfu.freq(key)
        assert freq_before == self.expect_freq_by_key[key]
        self.lfu.put(key, newval)
        if newval == self.expect_items[key]:  # same as old val -> put should have been a no-op
            assert self.lfu.freq(key) == freq_before
            return
        # new val different from old val -> freq of key should be reset to 0
        assert self.lfu.freq(key) == 0
        self.expect_freq_by_key[key] = 0
        self.expect_items[key] = newval

    @precondition(lambda self: self.expect_items)
    @rule(data=data())
    def get(self, data):
        key = data.draw(sampled_from(tuple(self.expect_items)))
        expect_val = self.expect_items.get(key)
        assert expect_val is not None
        freq_before = self.expect_freq_by_key[key]
        val = self.lfu.get(key)
        assert val == expect_val
        self.expect_freq_by_key[key] += 1
        assert self.lfu.freq(key) == freq_before + 1


Test = LFUCacheStateMachine.TestCase
