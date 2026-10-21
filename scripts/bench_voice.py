#!/usr/bin/env python3
"""Bounded v2 voice load probe. Requires aiohttp and opuslib_next (libopus).

Uses synthetic audio only; never sends credentials. Does not alter deployment.
Writes one record per attempted turn, including failed turns; no automatic retries.
"""
import argparse
import asyncio
import hashlib
import json
import math
import statistics
import struct
import time
import unicodedata
import wave
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import opuslib_next

TEXT = "Hôm nay trời khá nóng. Bạn nên mang theo ô và uống đủ nước khi ra ngoài."
EXPECTED = "Hôm nay ở Sài Gòn bao nhiêu độ, có nên mang ô không?"
AUDIO = {
    "input": {"codec": "opus", "sample_rate": 16000, "frame_ms": 20,
              "packet_ms": 20, "max_packet_bytes": 1275, "bitrate": 24000,
              "complexity": 3, "application": "restricted_lowdelay"},
    "output": {"codec": "opus", "sample_rate": 24000, "frame_ms": 20,
               "packet_ms": 20, "max_packet_bytes": 1275, "bitrate": 24000,
               "complexity": 3, "application": "audio"},
}


def norm(s):
    return " ".join("".join(c.lower() if c.isalnum() else " "
                            for c in unicodedata.normalize("NFC", s)).split())


def stats(values):
    if not values:
        return None
    values = sorted(values)
    return {"n": len(values), "median": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values)) - 1], "max": max(values)}


def summarize(rows):
    good = [r for r in rows if r.get("ok")]
    result = {"attempts": len(rows), "completed": len(good),
              "failed": len(rows) - len(good), "errors": {}}
    for r in rows:
        if not r.get("ok"):
            e = r.get("error", "unknown")
            result["errors"][e] = result["errors"].get(e, 0) + 1
    for k in ["connect_s", "stt_after_finish_s", "tts_first_audio_s",
              "speech_end_to_audio_s", "tts_rtf_including_queue", "tts_audio_s",
              "send_lag_max_s", "turn_s"]:
        result[k] = stats([r[k] for r in good if k in r])
    return result


async def event(ws):
    m = await ws.receive(timeout=135)
    if m.type == aiohttp.WSMsgType.BINARY:
        return m.data
    if m.type != aiohttp.WSMsgType.TEXT:
        raise RuntimeError("websocket_closed_" + str(m.type))
    d = json.loads(m.data)
    if d.get("type") == "error":
        raise RuntimeError("server_" + d.get("code", "error") + ":" + d.get("message", ""))
    return d


async def until(ws, kind):
    while True:
        d = await event(ws)
        if isinstance(d, dict) and d.get("type") == kind:
            return d
        if isinstance(d, dict) and d.get("type") == "request.end":
            raise RuntimeError("unexpected_request_end_" + json.dumps(d.get("result")))


class Probe:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=True)
        with wave.open(args.fixture, "rb") as f:
            assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (16000, 1, 2)
            raw = f.readframes(f.getnframes())
        self.duration = len(raw) / 32000
        enc = opuslib_next.Encoder(16000, 1, "restricted_lowdelay")
        enc.bitrate = 24000
        enc.complexity = 3
        self.packets = [enc.encode(raw[o:o+640].ljust(640, b"\0"), 320)
                        for o in range(0, len(raw), 640)]
        self.rows = []
        config = vars(args) | {"started_utc": datetime.now(timezone.utc).isoformat(),
                               "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
                               "audio_s": self.duration, "expected_transcript": EXPECTED,
                               "tts_text": TEXT, "audio": AUDIO,
                               "latency_budget_s": 3,
                               "method": "warm synchronized waves; no LLM; explicit STT finish; nearest-rank p95"}
        (self.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    def record(self, row):
        self.rows.append(row)
        with (self.out / "requests.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def turn(self, ws, n, wave_id, user, connect_s):
        begin = time.perf_counter()
        submitted = None
        submitted_stage = None
        r = {"concurrency": n, "wave": wave_id, "user": user,
             "profile": self.args.profile, "connect_s": connect_s,
             "started_utc": datetime.now(timezone.utc).isoformat()}
        try:
            async with asyncio.timeout(150):
                rid = f"b{n}w{wave_id}u{user}"
                if self.args.profile == "mixed":
                    await ws.send_json({"type": "stt.start", "request_id": rid + "s",
                                        "turn_id": rid, "end_silence_ms": 0})
                    ready = await until(ws, "stt.ready")
                    sid = ready["stream_id"]
                    start = time.perf_counter()
                    lags = []
                    for seq, packet in enumerate(self.packets):
                        target = start + seq * .020
                        await asyncio.sleep(max(0, target - time.perf_counter()))
                        lags.append(max(0, time.perf_counter() - target))
                        await ws.send_bytes(struct.pack("!II", sid, seq) + packet)
                    await asyncio.sleep(max(0, start + len(self.packets)*.020 - time.perf_counter()))
                    finish = time.perf_counter()
                    submitted, submitted_stage = finish, "stt"
                    await ws.send_json({"type": "stt.finish", "request_id": rid + "s"})
                    final = await until(ws, "stt.final")
                    r["stt_after_finish_s"] = time.perf_counter() - finish
                    r["transcript"] = final["text"]
                    r["transcript_exact_normalized"] = norm(final["text"]) == norm(EXPECTED)
                    end = await until(ws, "request.end")
                    if end["status"] != "completed":
                        raise RuntimeError("stt_" + json.dumps(end.get("result")))
                    if not r["transcript_exact_normalized"]:
                        raise RuntimeError("transcript_mismatch")
                    r["send_lag_max_s"] = max(lags)
                else:
                    finish = None

                await ws.send_json({"type": "tts.start", "request_id": rid + "t", "turn_id": rid})
                ready = await until(ws, "tts.ready")
                sid = ready["stream_id"]
                # Start timing when text becomes available, including transport and queueing.
                text_at = time.perf_counter()
                submitted, submitted_stage = text_at, "tts"
                await ws.send_json({"type": "tts.text", "request_id": rid + "t", "index": 0, "text": TEXT})
                await ws.send_json({"type": "tts.finish", "request_id": rid + "t"})
                packets = []
                first = None
                while True:
                    e = await event(ws)
                    if isinstance(e, bytes):
                        if first is None:
                            first = time.perf_counter()
                        stream, seq = struct.unpack("!II", e[:8])
                        if stream != sid or seq != len(packets):
                            raise RuntimeError("audio_sequence_mismatch")
                        packets.append(e[8:])
                    elif e.get("type") == "request.end":
                        if e["status"] != "completed":
                            raise RuntimeError("tts_" + json.dumps(e.get("result")))
                        if e.get("result", {}).get("packets") != len(packets):
                            raise RuntimeError("audio_packet_count_mismatch")
                        break
                last = time.perf_counter()
                if first is None or not packets:
                    raise RuntimeError("empty_audio")
                r["tts_first_audio_s"] = first - text_at
                r["tts_audio_s"] = len(packets) * .020
                r["tts_rtf_including_queue"] = (first - text_at) / r["tts_audio_s"]
                r["tts_delivery_s"] = last - first
                if finish is not None:
                    r["speech_end_to_audio_s"] = first - finish
                r["packets"] = len(packets)
                decoder = opuslib_next.Decoder(24000, 1)
                pcm = b"".join(decoder.decode(p, 480) for p in packets)
                if not any(pcm):
                    raise RuntimeError("silent_audio")
                if len(pcm) != len(packets) * 960:
                    raise RuntimeError("invalid_decoded_duration")
                r["ok"] = True
        except Exception as e:
            r["ok"] = False
            r["error"] = str(e) or type(e).__name__
            if submitted is not None and "overloaded" in r["error"]:
                r["overload_response_s"] = time.perf_counter() - submitted
                r["overload_stage"] = submitted_stage
        r["turn_s"] = time.perf_counter() - begin
        self.record(r)
        return r

    async def health(self, session):
        try:
            async with session.get(self.args.url.replace("wss:", "https:").replace("ws:", "http:").replace("/v2/voice", "/healthz"), timeout=aiohttp.ClientTimeout(total=10)) as r:
                return r.status == 200 and (await r.json()).get("ok") is True
        except Exception:
            return False

    async def level(self, session, n):
        # Keep WebSockets open across all turns at this level.
        async def connect(user):
            start = time.perf_counter()
            ws = await session.ws_connect(self.args.url, max_msg_size=10000, timeout=aiohttp.ClientWSTimeout(ws_close=5))
            await ws.send_json({"type": "hello", "version": 2, "audio": AUDIO})
            await until(ws, "ready")
            return ws, time.perf_counter() - start
        connections = await asyncio.gather(*(connect(i) for i in range(n)))
        start_index = len(self.rows)
        try:
            for w in range(self.args.waves):
                wave_rows = await asyncio.gather(*(self.turn(ws, n, w, i, delay)
                                                  for i, (ws, delay) in enumerate(connections)))
                print(json.dumps({"level": n, "wave": w, **summarize(wave_rows)}), flush=True)
                if not await self.health(session):
                    raise RuntimeError("health_probe_failed_stop")
                # Pacing option models listening to the full reply before the next utterance.
                pause = max((r.get("tts_audio_s", 0) for r in wave_rows), default=0) if self.args.playback_pacing else 0
                await asyncio.sleep(max(self.args.pause, pause + (1 if self.args.playback_pacing else 0)))
        finally:
            await asyncio.gather(*(ws.close() for ws, _ in connections))
        return summarize(self.rows[start_index:])

    async def run(self):
        result = {"levels": [], "stop_reason": None}
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=100), timeout=aiohttp.ClientTimeout(total=None)) as session:
            assert await self.health(session), "Initial health probe failed"
            # One unmeasured warm-up turn before collecting samples.
            ws = await session.ws_connect(self.args.url)
            await ws.send_json({"type": "hello", "version": 2, "audio": AUDIO})
            await until(ws, "ready")
            warm = await self.turn(ws, 1, -1, 0, 0)
            await ws.close()
            assert warm["ok"], "Warm-up failed: " + str(warm.get("error"))
            self.rows.clear()
            for n in self.args.levels:
                summary = await self.level(session, n)
                result["levels"].append({"concurrency": n, **summary})
                (self.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
                if not self.args.no_stop:
                    latency = summary.get("speech_end_to_audio_s") or summary.get("tts_first_audio_s")
                    if summary["failed"] / summary["attempts"] >= .05 or (latency and latency["p95"] > self.args.latency_stop):
                        result["stop_reason"] = f"At least 5% failures or p95 latency above {self.args.latency_stop}-second safety limit"
                        break
                await asyncio.sleep(3)
            result["post_health_ok"] = await self.health(session)
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (self.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--profile", choices=["mixed", "tts"], default="mixed")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16])
    p.add_argument("--waves", type=int, default=5)
    p.add_argument("--pause", type=float, default=2)
    p.add_argument("--latency-stop", type=float, default=10)
    p.add_argument("--playback-pacing", action="store_true")
    p.add_argument("--no-stop", action="store_true", help="Only use after inspecting previous measurements")
    args = p.parse_args()
    if not args.url.startswith(("wss://", "ws://")) or min(args.levels) < 1 or max(args.levels) > 32 or not 1 <= args.waves <= 50:
        p.error("Invalid URL or bounded load settings")
    asyncio.run(Probe(args).run())
