import unittest
from inno_medical_claims_guard import core


def claim(i, **kw):
    d = dict(claim_id="X%d" % i, center="H01", code="S01", patient="P%d" % i, day=i, qty=1, amount=100.0)
    d.update(kw); return d


class TestClaims(unittest.TestCase):
    def base(self):
        return [claim(i, amount=100.0 + (i % 5)) for i in range(60)]

    def test_normal_claims_low_score(self):
        sc = core.score(self.base())
        self.assertTrue(all(s["score"] < 0.5 for s in sc))

    def test_upcoding_flagged_with_reason(self):
        cl = self.base(); cl.append(claim(99, amount=900.0))
        s = core.score(cl)[-1]
        self.assertGreater(s["score"], 0.5); self.assertTrue(any("unit price" in r for r in s["reasons"]))

    def test_duplicate_flagged(self):
        cl = self.base(); cl.append(dict(cl[3], claim_id="DUP"))
        s = core.score(cl)[-1]
        self.assertTrue(any("already billed" in r for r in s["reasons"]))

    def test_auc_perfect_and_random(self):
        self.assertEqual(core.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(core.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_center_trend_sorted(self):
        cl = self.base(); sc = core.score(cl)
        tr = core.center_trend(cl, sc)
        self.assertEqual(tr, sorted(tr, key=lambda x: -x[1]))
