export const DEFAULT_MAX_QUEUED = 8;

export class InferenceQueueOverflowError extends Error {
  constructor() {
    super('The local detector is at capacity. This image was not analyzed.');
    this.name = 'InferenceQueueOverflowError';
    this.code = 'OFFSCREEN_QUEUE_FULL';
    this.retryable = true;
  }
}

export class SerialInferenceQueue {
  constructor({ maxQueued = DEFAULT_MAX_QUEUED } = {}) {
    if (!Number.isInteger(maxQueued) || maxQueued < 0) {
      throw new TypeError('maxQueued must be a non-negative integer.');
    }
    this.maxQueued = maxQueued;
    this.waiting = [];
    this.running = false;
  }

  get queued() {
    return this.waiting.length;
  }

  get active() {
    return this.running ? 1 : 0;
  }

  run(task) {
    if (typeof task !== 'function') return Promise.reject(new TypeError('task must be a function.'));
    if (this.running && this.waiting.length >= this.maxQueued) {
      return Promise.reject(new InferenceQueueOverflowError());
    }

    const promise = new Promise((resolve, reject) => {
      this.waiting.push({ task, resolve, reject });
    });
    this.drain();
    return promise;
  }

  drain() {
    if (this.running) return;
    const next = this.waiting.shift();
    if (!next) return;
    this.running = true;
    Promise.resolve()
      .then(next.task)
      .then(next.resolve, next.reject)
      .finally(() => {
        this.running = false;
        this.drain();
      });
  }
}
