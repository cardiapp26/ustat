import { afterEach, describe, expect, it, vi } from "vitest";
import { consumeLaunchedFiles } from "./fileLaunch";

type Consumer = (params: { files: readonly FileSystemHandle[] }) => void;

function installLaunchQueue(): { deliver: Consumer } {
  let consumer: Consumer | null = null;
  Object.defineProperty(window, "launchQueue", {
    configurable: true,
    value: { setConsumer: (c: Consumer) => { consumer = c; } },
  });
  return {
    deliver: (params) => {
      if (!consumer) throw new Error("no consumer set");
      consumer(params);
    },
  };
}

function fileHandle(file: File): FileSystemHandle {
  return { kind: "file", name: file.name, getFile: () => Promise.resolve(file) } as unknown as FileSystemHandle;
}

afterEach(() => {
  delete (window as { launchQueue?: unknown }).launchQueue;
});

describe("consumeLaunchedFiles", () => {
  it("is a no-op where the browser has no launchQueue", () => {
    const onFile = vi.fn();
    expect(() => consumeLaunchedFiles(onFile, vi.fn())).not.toThrow();
    expect(onFile).not.toHaveBeenCalled();
  });

  it("hands the launched file to onFile", async () => {
    const queue = installLaunchQueue();
    const onFile = vi.fn();
    consumeLaunchedFiles(onFile, vi.fn());
    const sav = new File(["x"], "cohort.sav");
    queue.deliver({ files: [fileHandle(sav)] });
    await vi.waitFor(() => expect(onFile).toHaveBeenCalledWith(sav));
  });

  it("opens only the first file of a multi-file launch", async () => {
    const queue = installLaunchQueue();
    const onFile = vi.fn();
    consumeLaunchedFiles(onFile, vi.fn());
    const first = new File(["a"], "a.csv");
    queue.deliver({ files: [fileHandle(first), fileHandle(new File(["b"], "b.xlsx"))] });
    await vi.waitFor(() => expect(onFile).toHaveBeenCalledTimes(1));
    expect(onFile).toHaveBeenCalledWith(first);
  });

  it("ignores a launch with no file, such as a plain app start", async () => {
    const queue = installLaunchQueue();
    const onFile = vi.fn();
    consumeLaunchedFiles(onFile, vi.fn());
    const dir = { kind: "directory", name: "data" } as unknown as FileSystemHandle;
    queue.deliver({ files: [] });
    queue.deliver({ files: [dir] });
    await Promise.resolve();
    expect(onFile).not.toHaveBeenCalled();
  });

  it("reports a file that can no longer be read", async () => {
    const queue = installLaunchQueue();
    const onFile = vi.fn();
    const onError = vi.fn();
    consumeLaunchedFiles(onFile, onError);
    const failure = new DOMException("gone", "NotFoundError");
    const broken = { kind: "file", name: "x.xls", getFile: () => Promise.reject(failure) } as unknown as FileSystemHandle;
    queue.deliver({ files: [broken] });
    await vi.waitFor(() => expect(onError).toHaveBeenCalledWith(failure));
    expect(onFile).not.toHaveBeenCalled();
  });
});

describe("a launch that arrives before the welcome screen mounts", () => {
  // Module state (the buffered launch, the launched-with-file flag) is what is
  // under test, so each case loads its own copy.
  async function freshModule() {
    vi.resetModules();
    return import("./fileLaunch");
  }

  it("is held until a listener asks for it", async () => {
    const queue = installLaunchQueue();
    const { watchFileLaunches, consumeLaunchedFiles: consume } = await freshModule();
    watchFileLaunches();
    const sav = new File(["x"], "cohort.sav");
    queue.deliver({ files: [fileHandle(sav)] });

    const onFile = vi.fn();
    consume(onFile, vi.fn());
    await vi.waitFor(() => expect(onFile).toHaveBeenCalledExactlyOnceWith(sav));
  });

  it("is delivered once, not again to a later listener", async () => {
    const queue = installLaunchQueue();
    const { watchFileLaunches, consumeLaunchedFiles: consume } = await freshModule();
    watchFileLaunches();
    queue.deliver({ files: [fileHandle(new File(["x"], "a.csv"))] });

    const first = vi.fn();
    const stop = consume(first, vi.fn());
    await vi.waitFor(() => expect(first).toHaveBeenCalledTimes(1));
    stop();
    const second = vi.fn();
    consume(second, vi.fn());
    await Promise.resolve();
    expect(second).not.toHaveBeenCalled();
  });

  it("stops delivering to a listener that unsubscribed", async () => {
    const queue = installLaunchQueue();
    const { consumeLaunchedFiles: consume } = await freshModule();
    const onFile = vi.fn();
    consume(onFile, vi.fn())();
    queue.deliver({ files: [fileHandle(new File(["x"], "a.csv"))] });
    await Promise.resolve();
    expect(onFile).not.toHaveBeenCalled();
  });

  it("records that the app was launched with a file", async () => {
    const queue = installLaunchQueue();
    const { watchFileLaunches, wasLaunchedWithFile } = await freshModule();
    watchFileLaunches();
    expect(wasLaunchedWithFile()).toBe(false);
    queue.deliver({ files: [] });
    expect(wasLaunchedWithFile()).toBe(false);
    queue.deliver({ files: [fileHandle(new File(["x"], "a.csv"))] });
    expect(wasLaunchedWithFile()).toBe(true);
  });

  it("reports no file launch where the browser has no launchQueue", async () => {
    const { watchFileLaunches, wasLaunchedWithFile } = await freshModule();
    expect(() => watchFileLaunches()).not.toThrow();
    expect(wasLaunchedWithFile()).toBe(false);
  });
});
