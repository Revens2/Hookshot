"""RX_VK_IMPORT_LOAD_MB imports large ranges at load and keeps the mirror at today's granularity (#3013; no game/GPU required).
A load-sized import is split, off the replay thread, into one import per mirror chunk a game pool covers; each part is mirrored
at its first binding as any chunk, so the mirrors and the lookups stay the 4 MB ones. Off by default: import_chunk() is then
the mirror chunk, unchanged.

python -I -B tests/test_import_load_split.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def block(text, marker):
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def need(haystack, needle, what):
    if needle not in haystack:
        raise SystemExit("vk_memory.cpp: " + what)


def main():
    mem = (ROOT / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    load = block(mem, "uint64_t import_chunk() {")
    need(load, 'getenv("RX_VK_IMPORT_LOAD_MB")', "import_chunk() no longer reads RX_VK_IMPORT_LOAD_MB")
    need(load, "mirror_chunk()", "import_chunk() no longer starts from the mirror chunk")
    need(load, "1024", "import_chunk() no longer clamps the load range (the 1 GB no import exceeds)")
    need(load, "std::max(mirror, load)", "import_chunk() no longer takes the larger of the mirror chunk and the load range")

    parts = block(mem, "std::vector<std::pair<uint64_t, uint64_t>> split_parts(uint64_t lo, uint64_t hi, uint64_t chunk) {")
    need(parts, "g.lo / chunk * chunk", "split_parts() no longer rounds a game pool down to the mirror chunk")
    need(parts, "(g.hi + chunk - 1) / chunk * chunk", "split_parts() no longer rounds a game pool's end up to the mirror chunk")
    need(parts, "std::min(hi,", "split_parts() no longer keeps the parts inside the import")

    split = block(mem, "bool split_import(Pool& p) {")
    need(split, "mirror_on()", "split_import() no longer leaves pools without mirrors alone")
    need(split, "whole <= chunk", "split_import() no longer requires a load-sized import (imports at the mirror chunk are not split)")
    need(split, "split_parts(p.lo, p.hi, chunk)", "split_import() no longer splits by the mirror chunk")
    need(split, "p.splitting = true", "split_import() no longer marks the pool splitting")
    need(split, "p.mstate = Pool::MIRROR_REFUSED", "split_import() no longer keeps the large import out of VRAM meanwhile")
    need(split, "p.split_id = split_next_id++", "split_import() no longer tags the split (a stale result must not apply to a new import)")
    need(split, "split_queue.push_back(", "split_import() no longer queues the parts on the background thread")

    lookup = block(mem, "Pool* pool_lookup(uint64_t guest, uint64_t size) {")
    need(lookup, "in_split_region(guest, guest + size)", "pool_lookup() no longer looks up at the mirror chunk inside a split range")
    need(lookup, "chunk = mirror_chunk()", "pool_lookup() no longer falls back to the mirror chunk inside a split range")
    need(lookup, "split_import(imp->second)", "pool_lookup() no longer splits a load-sized import")

    background = block(mem, "bool import_in_background(uint64_t guest, uint64_t size) {")
    need(background, "in_split_region(guest, guest + size)", "import_in_background() no longer looks up at the mirror chunk inside a split range")

    drain = block(mem, "void split_drain() {")
    need(drain, "it->second.split_id == s.split.id", "split_drain() no longer drops a result for an import it was not made for")
    need(drain, "retire_pool(it)", "split_drain() no longer retires the large import")
    need(drain, "split_regions.push_back({s.split.lo, s.split.hi})", "split_drain() no longer records the range as split")
    need(drain, "guest_pages_set(ins.first->second.lo", "split_drain() no longer publishes the parts' imports to the page table")
    need(drain, 'imported.try_emplace(p.lo, std::move(p))', "split_drain() no longer takes the part's row in the table")
    need(drain, "free_import(p)", "split_drain() no longer gives back the parts it cannot use")
    if "MIRROR_WANTED" in drain:
        raise SystemExit("vk_memory.cpp: split_drain() makes the parts' mirrors at once: they must be made at their first "
                         "binding, as any chunk's (mirror granularity and bytes unchanged)")

    main_queue = block(mem, "void bg_main() {")
    need(main_queue, "split_queue", "bg_main() no longer imports the split's parts")
    need(main_queue, "split_ready.push_back", "bg_main() no longer hands the split's imports over")

    bg = block(mem, "void bg_drain() {")
    need(bg, "split_drain()", "bg_drain() no longer applies finished splits")
    need(bg, "split_import(ins->second)", "bg_drain() no longer splits a load-sized import the thread made")

    trim = block(mem, "bool mirrors_trim() {")
    need(trim, "mirror_chunk()", "mirrors_trim() no longer re-imports a trim's parts at the mirror chunk")
    if "import_chunk()" in trim:
        raise SystemExit("vk_memory.cpp: mirrors_trim() re-imports at the import's size: with RX_VK_IMPORT_LOAD_MB a trim "
                         "would bring a load-sized range and its mirror back")

    bind = block(mem, "BufferRef gpu_buffer_bind(")
    if bind.count("p->splitting") < 2:
        raise SystemExit("vk_memory.cpp: gpu_buffer_bind() no longer serves a splitting import's bindings in guest memory "
                         "(both its cached and its locked path)")

    make = block(mem, "bool mirror_make(Pool& pool) {")
    need(make, "pool.splitting", "mirror_make() no longer refuses a mirror of a splitting import")

    settle = block(mem, "void trims_settle() {")
    need(settle, "!q.splitting", "trims_settle() can turn a splitting import back to MIRROR_NONE (it would then take a "
                                 "whole-range mirror while its parts are being made)")

    print("load-sized imports split to the mirror chunk, off the replay thread: ok")


if __name__ == "__main__":
    main()