"""Zero-dependency TensorBoard event reader: TFRecord frames + protobuf wire.

Walks a logdir, parses every events.out.tfevents.* file, extracts (tag, step,
simple_value) triples, groups by tag, prints a downsampled trend per tag.
"""
import struct, sys, glob, os
from collections import defaultdict


def _varint(buf, i):
    shift = 0; result = 0
    while True:
        b = buf[i]; i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _fields(buf):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i); yield fn, v
        elif wt == 1:
            yield fn, buf[i:i+8]; i += 8
        elif wt == 2:
            ln, i = _varint(buf, i); yield fn, buf[i:i+ln]; i += ln
        elif wt == 5:
            yield fn, buf[i:i+4]; i += 4
        else:
            raise ValueError(f"wire type {wt}")


def _records(path):
    with open(path, "rb") as f:
        data = f.read()
    i, n = 0, len(data)
    while i + 12 <= n:
        length = struct.unpack("<Q", data[i:i+8])[0]
        i += 12  # 8 length + 4 length-crc
        if i + length + 4 > n:
            break
        yield data[i:i+length]
        i += length + 4  # data + 4 data-crc


def parse_logdir(logdir):
    series = defaultdict(list)  # tag -> [(step, value)]
    files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    for path in files:
        try:
            for rec in _records(path):
                step = None; summ = None
                for fn, val in _fields(rec):
                    if fn == 2 and isinstance(val, int):
                        step = val
                    elif fn == 5 and isinstance(val, (bytes, bytearray)):
                        summ = val
                if summ is None:
                    continue
                for fn, val in _fields(summ):
                    if fn != 1 or not isinstance(val, (bytes, bytearray)):
                        continue
                    tag = None; sv = None
                    for vfn, vval in _fields(val):
                        if vfn == 1 and isinstance(vval, (bytes, bytearray)):
                            tag = vval.decode("utf-8", "replace")
                        elif vfn == 2 and isinstance(vval, (bytes, bytearray)) and len(vval) == 4:
                            sv = struct.unpack("<f", vval)[0]
                    if tag is not None and sv is not None and step is not None:
                        series[tag].append((step, sv))
        except Exception as e:
            print(f"  (skip {os.path.basename(path)}: {e})")
    for tag in series:
        series[tag].sort()
    return series


def fmt_trend(pairs, n=14):
    if not pairs:
        return "(none)"
    if len(pairs) <= n:
        idx = range(len(pairs))
    else:
        step = (len(pairs) - 1) / (n - 1)
        idx = [int(round(k * step)) for k in range(n)]
    return "  ".join(f"{pairs[i][0]//1000}k:{pairs[i][1]:.3g}" for i in idx)


if __name__ == "__main__":
    logdir = sys.argv[1]
    series = parse_logdir(logdir)
    want = ["average_episode_rewards", "value_loss", "explained_variance",
            "dist_entropy", "policy_loss", "ratio", "approx_kl", "clip_fraction",
            "actor_grad_norm", "critic_grad_norm",
            "mec/accepted", "mec/U_src", "mec/overflow", "mec/w1",
            "mec/training_cost", "mec/src_cost", "mec/ovf_cost"]
    print(f"tags found: {len(series)};  points in reward: {len(series.get('average_episode_rewards', []))}\n")
    for tag in want:
        if tag in series:
            print(f"{tag:24s} {fmt_trend(series[tag])}")
    extra = [t for t in series if t not in want and not t.startswith("agent")]
    if extra:
        print("\nother tags:", extra)
