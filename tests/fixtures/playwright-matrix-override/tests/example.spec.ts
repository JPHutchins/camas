import { test, expect } from "@playwright/test";

test("homepage has title @smoke", async ({ page }) => {
	await page.goto("https://example.com/");
	await expect(page).toHaveTitle(/Example/);
});
