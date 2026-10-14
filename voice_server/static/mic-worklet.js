class MicrophoneProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.previous = 0;
    this.inputIndex = 0;
    this.nextOutputIndex = 0;
    this.targetRate = options.processorOptions?.sampleRate ?? 48000;
    this.frameSamples = this.targetRate * (options.processorOptions?.frameMs ?? 20) / 1000;
    this.ratio = sampleRate / this.targetRate;
    this.frame = new Int16Array(this.frameSamples);
    this.frameOffset = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    if (input) {
      for (const sample of input) {
        while (this.nextOutputIndex <= this.inputIndex) {
          const fraction = this.nextOutputIndex - (this.inputIndex - 1);
          const value = this.previous + (sample - this.previous) * fraction;
          this.frame[this.frameOffset++] = Math.round(Math.max(-1, Math.min(1, value)) * 32767);
          if (this.frameOffset === this.frame.length) {
            this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
            this.frame = new Int16Array(this.frameSamples);
            this.frameOffset = 0;
          }
          this.nextOutputIndex += this.ratio;
        }
        this.previous = sample;
        this.inputIndex++;
      }
    }
    for (const output of outputs) {
      for (const channel of output) channel.fill(0);
    }
    return true;
  }
}

registerProcessor("microphone-processor", MicrophoneProcessor);
