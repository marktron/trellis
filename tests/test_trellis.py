"""Unit tests for trellis pure helpers. Run with:
    python3 /Users/mark/Developer/trellis/tests/test_trellis.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import trellis as t  # noqa: E402


class TestFrontmatter(unittest.TestCase):
    def test_no_frontmatter(self):
        fm, body = t.split_frontmatter("# Title\n\nhello")
        self.assertEqual(fm, "")
        self.assertEqual(body, "# Title\n\nhello")

    def test_split_and_body(self):
        text = "---\ntags:\n  - a\n  - b\n---\nbody here"
        fm, body = t.split_frontmatter(text)
        self.assertIn("tags:", fm)
        self.assertEqual(body, "body here")

    def test_tags_indented_list(self):
        fm = "tags:\n  - cycling/randonneuring\n  - health\nfoo: bar"
        self.assertEqual(t.extract_tags(fm), ["cycling/randonneuring", "health"])

    def test_tags_inline_list(self):
        fm = "tags: [moc, strategy]"
        self.assertEqual(t.extract_tags(fm), ["moc", "strategy"])

    def test_tags_absent(self):
        self.assertEqual(t.extract_tags("title: hi\nfoo: bar"), [])


class TestHashAndText(unittest.TestCase):
    def test_hash_deterministic(self):
        self.assertEqual(t.content_hash(b"abc"), t.content_hash(b"abc"))
        self.assertNotEqual(t.content_hash(b"abc"), t.content_hash(b"abd"))

    def test_build_embed_text_includes_parts(self):
        out = t.build_embed_text("My Note", ["x", "y"], "the body", 1000)
        self.assertIn("My Note", out)
        self.assertIn("tags: x, y", out)
        self.assertIn("the body", out)

    def test_build_embed_text_truncates(self):
        out = t.build_embed_text("T", [], "z" * 100, 10)
        self.assertEqual(len(out), 10)


class TestVectorMath(unittest.TestCase):
    def test_normalize_unit_length(self):
        m = t.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(m[0])), 1.0, places=5)
        self.assertEqual(float(np.linalg.norm(m[1])), 0.0)  # zero vector stays zero

    def test_top_k_ranking(self):
        mat = t.l2_normalize(np.array([
            [1.0, 0.0],   # 0 — identical to query
            [0.9, 0.1],   # 1 — close
            [0.0, 1.0],   # 2 — orthogonal
        ], dtype=np.float32))
        q = np.array([1.0, 0.0], dtype=np.float32)
        ranked = t.top_k(q, mat, k=3)
        self.assertEqual([i for i, _ in ranked], [0, 1, 2])
        self.assertGreaterEqual(ranked[0][1], ranked[1][1])

    def test_top_k_empty(self):
        self.assertEqual(t.top_k(np.array([1.0]), np.zeros((0, 1), np.float32), 5), [])


class TestGardenGraph(unittest.TestCase):
    def test_parse_outlinks(self):
        body = "see [[Note A]] and [[Note B|alias]] and [[Note C#heading]] and [[Note A]]"
        self.assertEqual(t.parse_outlinks(body), {"note a", "note b", "note c"})

    def test_parse_outlinks_none(self):
        self.assertEqual(t.parse_outlinks("plain text, no links"), set())

    def test_build_graph_and_inbound(self):
        notes = {
            "z/a.md": {"title": "A", "out": {"b"}},
            "z/b.md": {"title": "B", "out": {"a", "ghost"}},   # ghost doesn't exist
            "z/c.md": {"title": "C", "out": set()},
        }
        t2r, inbound = t.build_link_graph(notes)
        self.assertEqual(t2r["a"], "z/a.md")
        self.assertEqual(inbound["z/a.md"], 1)   # b -> a
        self.assertEqual(inbound["z/b.md"], 1)   # a -> b
        self.assertEqual(inbound["z/c.md"], 0)   # nobody links to c
        self.assertNotIn("ghost", t2r)           # unresolved target ignored

    def test_resolved_outlinks_drops_unresolved_and_self(self):
        notes = {"z/a.md": {"title": "A", "out": {"a", "b", "ghost"}},
                 "z/b.md": {"title": "B", "out": set()}}
        t2r, _ = t.build_link_graph(notes)
        self.assertEqual(t.resolved_outlinks(notes["z/a.md"], "z/a.md", t2r), {"b"})

    def test_orphan_detection(self):
        notes = {"z/a.md": {"title": "A", "out": set()},
                 "z/b.md": {"title": "B", "out": {"a"}}}
        t2r, inbound = t.build_link_graph(notes)
        # a has inbound (from b) -> not orphan; b has outbound -> not orphan
        self.assertEqual(inbound["z/a.md"], 1)
        self.assertEqual(t.resolved_outlinks(notes["z/b.md"], "z/b.md", t2r), {"a"})


class TestCandidateTags(unittest.TestCase):
    def test_frequency_ranked_excluding_own(self):
        neighbors = [["ai", "ethics"], ["ai", "politics"], ["ai", "ethics"]]
        out = t.candidate_tags(neighbors, own_tags={"politics"}, top_n=10)
        self.assertEqual(out[0], "ai")          # most frequent
        self.assertIn("ethics", out)
        self.assertNotIn("politics", out)       # excluded (already on note)

    def test_top_n_cap(self):
        neighbors = [["a", "b", "c", "d"]]
        self.assertEqual(len(t.candidate_tags(neighbors, set(), top_n=2)), 2)


class TestClassifyTagSuggestions(unittest.TestCase):
    def test_existing_vault_tag_not_labeled_new(self):
        # "health" isn't in this note's neighbor candidates, but it exists in the
        # vault — it must NOT be returned as a new tag (the reported bug).
        picked, new_tag = t.classify_tag_suggestions(
            picked_raw=[], proposed_raw=["health"],
            cand_tags=["ai"], vault_tags={"health", "ai", "cycling"}, own_tags=[])
        self.assertEqual(new_tag, [])
        self.assertIn("health", picked)   # reclassified as a normal pick

    def test_genuinely_new_tag_kept(self):
        picked, new_tag = t.classify_tag_suggestions(
            picked_raw=[], proposed_raw=["quantum-foo"],
            cand_tags=["ai"], vault_tags={"ai"}, own_tags=[])
        self.assertEqual(new_tag, ["quantum-foo"])
        self.assertEqual(picked, [])

    def test_existing_match_is_case_insensitive(self):
        picked, new_tag = t.classify_tag_suggestions(
            picked_raw=[], proposed_raw=["Health"],
            cand_tags=[], vault_tags={"health"}, own_tags=[])
        self.assertEqual(new_tag, [])
        self.assertIn("Health", picked)   # original case preserved

    def test_own_tag_dropped_not_suggested(self):
        # a tag already on the note is neither "new" nor a fresh pick
        picked, new_tag = t.classify_tag_suggestions(
            picked_raw=[], proposed_raw=["health"],
            cand_tags=[], vault_tags={"health"}, own_tags=["health"])
        self.assertEqual(new_tag, [])
        self.assertNotIn("health", picked)

    def test_picked_filtered_to_candidates(self):
        picked, _ = t.classify_tag_suggestions(
            picked_raw=["ai", "bogus"], proposed_raw=[],
            cand_tags=["ai"], vault_tags={"ai"}, own_tags=[])
        self.assertEqual(picked, ["ai"])   # hallucinated "bogus" dropped

    def test_handles_none_inputs(self):
        # some models emit {"tags": null, "proposed_new": null}; dict.get returns
        # the stored None (not the default), so the helper must coerce safely.
        picked, new_tag = t.classify_tag_suggestions(
            picked_raw=None, proposed_raw=None,
            cand_tags=["ai"], vault_tags={"ai"}, own_tags=[])
        self.assertEqual(picked, [])
        self.assertEqual(new_tag, [])

    def test_new_tag_capped_to_one(self):
        _, new_tag = t.classify_tag_suggestions(
            picked_raw=[], proposed_raw=["new-a", "new-b"],
            cand_tags=[], vault_tags=set(), own_tags=[])
        self.assertEqual(len(new_tag), 1)


class TestRenderReport(unittest.TestCase):
    def test_render_includes_sections_and_checkboxes(self):
        md = t.render_report(
            "2026-06-15",
            {"processed": 2, "new_links": 1, "new_tags": 1, "orphans": 5},
            link_items=[{"source": "Note A",
                         "suggestions": [{"title": "Note B", "reason": "shared theme"}]}],
            tag_items=[{"source": "Note A", "tags": ["ai"], "proposed_new": []}],
            orphans=["z/x.md"])
        self.assertIn("# Gardener review — 2026-06-15", md)
        self.assertIn("- [ ] link → [[Note B]] — shared theme", md)
        self.assertIn("`ai`", md)
        self.assertIn("Orphans in scope", md)

    def test_render_empty(self):
        md = t.render_report("2026-06-15",
                             {"processed": 0, "new_links": 0, "new_tags": 0, "orphans": 0},
                             [], [], [])
        self.assertIn("No new suggestions", md)


class TestParseReview(unittest.TestCase):
    SAMPLE = """# Gardener review — 2026-06-15

## Link suggestions

### [[Note A]]
- [x] link → [[Target One]] — good reason
- [ ] link → [[Target Two]] — unchecked, ignore

### [[Note B]]
- [x] link → [[Target Three]] — keep

## Tag suggestions

- [x] [[Note A]] → `health` `weight`
- [ ] [[Note C]] → `ignored`

## Orphans in scope (5 total)

- [[Some Orphan]]
"""

    def test_links_only_checked(self):
        r = t.parse_review(self.SAMPLE)
        self.assertIn(("Note A", "Target One"), r["links"])
        self.assertIn(("Note B", "Target Three"), r["links"])
        self.assertNotIn(("Note A", "Target Two"), r["links"])  # unchecked

    def test_tags_checked_with_multiple(self):
        r = t.parse_review(self.SAMPLE)
        self.assertEqual(r["tags"], [("Note A", ["health", "weight"])])

    def test_orphans_section_ignored(self):
        # bare [[links]] in the orphans section must not become suggestions
        r = t.parse_review(self.SAMPLE)
        flat = [tgt for _, tgt in r["links"]]
        self.assertNotIn("Some Orphan", flat)


class TestArchiveReview(unittest.TestCase):
    def test_moves_file_into_applied_subdir(self):
        import tempfile
        d = tempfile.mkdtemp()
        f = os.path.join(d, "2026-06-16.md")
        with open(f, "w") as fh:
            fh.write("review")
        dest = t._archive_review(f)
        self.assertEqual(dest, os.path.join(d, "applied", "2026-06-16.md"))
        self.assertFalse(os.path.exists(f))       # original moved
        self.assertTrue(os.path.exists(dest))     # now in applied/

    def test_does_not_clobber_existing_archive(self):
        import tempfile
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "applied"))
        with open(os.path.join(d, "applied", "r.md"), "w") as fh:
            fh.write("old")
        f = os.path.join(d, "r.md")
        with open(f, "w") as fh:
            fh.write("new")
        dest = t._archive_review(f)
        self.assertNotEqual(os.path.basename(dest), "r.md")   # timestamped
        with open(os.path.join(d, "applied", "r.md")) as fh:
            self.assertEqual(fh.read(), "old")                # prior archive intact


class TestGeneratePayload(unittest.TestCase):
    def test_caps_tokens_disables_thinking_and_passes_timeout(self):
        captured = {}

        def fake_post(url, payload, timeout=120.0):
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {"response": '{"links": []}'}

        orig = t._post
        t._post = fake_post
        try:
            out = t.generate_json("p", "m", "http://x", timeout=42, num_predict=256)
        finally:
            t._post = orig
        self.assertEqual(out, {"links": []})
        self.assertEqual(captured["payload"]["options"]["num_predict"], 256)
        self.assertEqual(captured["payload"]["options"]["temperature"], 0)
        self.assertFalse(captured["payload"]["think"])
        self.assertEqual(captured["timeout"], 42)

    def test_bad_json_returns_empty(self):
        orig = t._post
        t._post = lambda url, payload, timeout=120.0: {"response": "not json{"}
        try:
            self.assertEqual(t.generate_json("p", "m", "http://x"), {})
        finally:
            t._post = orig


class TestClusterHelpers(unittest.TestCase):
    def test_cluster_members_groups_and_drops_noise(self):
        labels = [0, 1, 0, -1, 1, 1]
        out = t.cluster_members(labels)
        self.assertEqual(out[0], [0, 2])
        self.assertEqual(out[1], [1, 4, 5])
        self.assertNotIn(-1, out)   # noise dropped

    def test_centroid_is_normalized_mean(self):
        mat = t.l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        c = t.centroid(mat, [0, 1])
        self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=5)
        self.assertAlmostEqual(float(c[0]), float(c[1]), places=5)  # symmetric

    def test_rank_by_centrality_orders_by_cosine(self):
        mat = t.l2_normalize(np.array([
            [1.0, 0.0],   # 0 — on axis
            [0.0, 1.0],   # 1 — orthogonal
            [0.9, 0.1],   # 2 — near axis
        ], dtype=np.float32))
        cen = np.array([1.0, 0.0], dtype=np.float32)
        self.assertEqual(t.rank_by_centrality(mat, [0, 1, 2], cen), [0, 2, 1])

    def test_coverage_score_picks_nearest_moc(self):
        moc = t.l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        cen = np.array([1.0, 0.0], dtype=np.float32)
        j, score = t.coverage_score(cen, moc)
        self.assertEqual(j, 0)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_coverage_score_no_mocs(self):
        cen = np.array([1.0, 0.0], dtype=np.float32)
        self.assertEqual(t.coverage_score(cen, np.zeros((0, 2), np.float32)), (-1, 0.0))

    def test_moc_linked_targets_only_from_mocs(self):
        notes = {
            "MOCs/Strategy MOC.md": {"title": "Strategy MOC", "out": {"moats"}},
            "z/moats.md": {"title": "moats", "out": set()},
            "z/other.md": {"title": "other", "out": {"moats"}},  # not a MOC source
        }
        t2r, _ = t.build_link_graph(notes)
        linked = t.moc_linked_targets(notes, t2r)
        self.assertIn("z/moats.md", linked)          # linked from the MOC
        self.assertEqual(len(linked), 1)             # the z/ source doesn't count

    def test_link_coverage_fraction(self):
        self.assertEqual(t.link_coverage(["a", "b", "c", "d"], {"a", "c"}), 0.5)
        self.assertEqual(t.link_coverage([], {"a"}), 0.0)

    def test_filter_unseen_drops_known_anchors(self):
        cands = [{"anchor": "z/a.md"}, {"anchor": "z/b.md"}, {"anchor": "z/c.md"}]
        out = t.filter_unseen(cands, {"z/b.md"})
        self.assertEqual([c["anchor"] for c in out], ["z/a.md", "z/c.md"])

    def test_naming_prompt_includes_tags_and_titles(self):
        p = t.build_cluster_naming_prompt(["health", "aging"], ["Sleep and aging", "VO2max"])
        self.assertIn("health, aging", p)
        self.assertIn("Sleep and aging", p)
        self.assertIn("VO2max", p)
        self.assertIn("theme", p)          # asks for the JSON theme field

    def _sample_candidate(self):
        return {"anchor": "z/sleep.md", "theme": "Sleep & Aging", "tag": "aging/sleep",
                "rationale": "sleep quality declines with age", "member_count": 9,
                "link_coverage": 0.0, "nearest_moc": ("Active Aging & Longevity MOC", 0.41),
                "repr_titles": ["Sleep and aging", "Deep sleep"],
                "member_titles": ["Sleep and aging", "Deep sleep", "Naps"]}

    def test_render_report_includes_candidate_and_moc_line(self):
        md = t.render_cluster_report(
            "2026-06-19", {"clusters": 5, "candidates": 1, "covered": 4},
            [self._sample_candidate()])
        self.assertIn("# MOC candidates — 2026-06-19", md)
        self.assertIn("## Sleep & Aging", md)
        self.assertIn("`aging/sleep`", md)
        self.assertIn("/moc Sleep & Aging", md)
        self.assertIn("[[Sleep and aging]]", md)
        self.assertIn("Active Aging & Longevity MOC", md)

    def test_render_cluster_report_empty(self):
        md = t.render_cluster_report(
            "2026-06-19", {"clusters": 0, "candidates": 0, "covered": 0}, [])
        self.assertIn("No new MOC candidates", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
