# playwright-matrix-override

Fixture demonstrating CLI matrix overrides on a 2-axis matrix
(`BROWSER × VIEWPORT`). The Playwright shape is incidental — what's
shown is how `--AXIS` flags scope a matrix run from the command line:

```
camas e2e                                       # all 6: 3 browsers × 2 viewports
camas e2e --BROWSER chromium                    # 2: chromium × {desktop, mobile}
camas e2e --VIEWPORT mobile                     # 3: {chromium, firefox, webkit} × mobile
camas e2e --BROWSER chromium --VIEWPORT mobile  # 1: single combo
camas e2e --BROWSER chromium,firefox            # 4: two browsers × both viewports
```

`camas e2e --help` lists the available axes and their current values.
