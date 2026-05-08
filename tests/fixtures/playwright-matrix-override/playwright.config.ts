import { defineConfig, devices } from "@playwright/test";

const browsers = ["chromium", "firefox", "webkit"] as const;
const viewports = {
	desktop: devices["Desktop Chrome"],
	mobile: devices["Pixel 5"],
} as const;

export default defineConfig({
	testDir: "./tests",
	projects: browsers.flatMap((browser) =>
		(Object.keys(viewports) as Array<keyof typeof viewports>).map((viewport) => ({
			name: `${browser}-${viewport}`,
			use: { ...viewports[viewport], browserName: browser },
		})),
	),
});
