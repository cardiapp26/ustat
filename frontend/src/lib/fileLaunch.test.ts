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
