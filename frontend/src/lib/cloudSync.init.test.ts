/**
 * cloudSync.init() restores an expired Drive token with a silent full-page
 * redirect to Google. That navigation must not happen on a page the OS opened
 * with a file: it unloads the document holding the launched file, so the first
 * "Open With" after the token expires would open nothing.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./cloudConfig", async (importOriginal) => {
  const original = await importOriginal<typeof import("./cloudConfig")>();
  return { ...original, CLOUD_CONFIG: { ...original.CLOUD_CONFIG, GOOGLE_CLIENT_ID: "test-client-id" } };
});

const realLocation = window.location;
let navigatedTo: string | null = null;

beforeEach(() => {
  vi.resetModules();
  localStorage.clear();
  navigatedTo = null;
  // jsdom cannot navigate; capture the assignment instead.
  Object.defineProperty(window, "location", {
    configurable: true,
    value: {
      origin: realLocation.origin,
      pathname: "/",
      search: "",
      hash: "",
      get href() { return realLocation.href; },
      set href(url: string) { navigatedTo = url; },
    },
  });
  // A returning Drive user whose cached token has expired.
  localStorage.setItem("ustat_cloud_user", JSON.stringify({ email: "a@example.com", name: "A" }));
  localStorage.setItem("ustat_cloud_mode", "redirect");
  localStorage.setItem("ustat_cloud_token", JSON.stringify({ t: "stale", e: Date.now() - 1000 }));
});

afterEach(() => {
  Object.defineProperty(window, "location", { configurable: true, value: realLocation });
});

describe("cloudSync.init with an expired token", () => {
  it("re-authenticates with a silent redirect on an ordinary page load", async () => {
    const { cloudSync } = await import("./cloudSync");
    await cloudSync.init();
    expect(navigatedTo).toContain("accounts.google.com");
    expect(navigatedTo).toContain("prompt=none");
  });

  it("stays on the page when a redirect is not allowed", async () => {
    const { cloudSync } = await import("./cloudSync");
    await cloudSync.init({ allowRedirect: false });
    expect(navigatedTo).toBeNull();
    expect(cloudSync.isSignedIn()).toBe(false);
    expect(cloudSync.getStatus()).toMatchObject({ status: "idle", message: "Re-sign-in required" });
  });
});
