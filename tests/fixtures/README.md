# Endpointing fixture

`vi_weather_two_clauses.wav` is generated Vietnamese speech, converted to
16 kHz mono PCM16 without changing its timing. It contains no recorded person
or credentials. Duration: 4.85 seconds.

Spoken text: “Hôm nay ở Sài Gòn bao nhiêu độ, có nên mang ô không?”

A diagnostic silence detector at −35 dB found the clause pause from 2.756 to
3.442 seconds (686 ms). The old 512 ms endpoint finalized the first clause
before the second began. This fixture checks complete-utterance retention;
it is not a microphone, accent, noise or listening-quality acceptance test.
