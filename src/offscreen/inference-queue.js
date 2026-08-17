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

  // `priority` orders the waiting list: lower runs sooner. Ties keep
  // submission order (a stable insert, not a stable sort library) so
  // requests with no position information (or all equal) behave exactly
  // like the plain FIFO this queue used to be. A missing/invalid priority
  // sorts last, never blocking a request that did supply one.
  run(task, priority) {
    if (typeof task !== 'function') return Promise.reject(new TypeError('task must be a function.'));
    if (this.running && this.waiting.length >= this.maxQueued) {
      return Promise.reject(new InferenceQueueOverflowError());
    }

    const orderedPriority = Number.isFinite(priority) ? priority : Number.POSITIVE_INFINITY;
    const promise = new Promise((resolve, reject) => {
      const entry = { task, priority: orderedPriority, resolve, reject };
      let index = this.waiting.length;
      while (index > 0 && this.waiting[index - 1].priority > entry.priority) index -= 1;
      this.waiting.splice(index, 0, entry);
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
