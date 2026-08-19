# Course branding assets

The site uses University of Alberta evergreen `#275D38`, dark evergreen `#1C4630`, and gold `#F2CD00`.

- `ua-logo-reversed-white.svg` and `ua-logo-green.svg` are the English UAlberta marks referenced by the university website’s `framework-v2.css` as `_assets/images/ua-logo-reversed-white.svg` and `_assets/images/ua-logo-green.svg`.
- `uofa-crest.svg` is the university website favicon and supports light and dark colour schemes.
- `engineering-logo.png` is the official Faculty of Engineering raster mark supplied for this course.

The UAlberta website implements `<a class="navbar-brand en-logo">` as an empty link with a CSS background image. On a standard green faculty header, its relevant rules are effectively:

```css
.navbar-brand {
  background: url("../images/ua-logo-green.svg") no-repeat;
  background-size: contain;
  width: 183px;
  height: 50px;
}

.blade-wrapper.standard .navbar-brand {
  background-image: url("../images/ua-logo-reversed-white.svg");
}
```

MATE 374 uses a normal Quarto `<img>` with the same white SVG rather than recreating UAlberta’s complete header framework. This preserves accessibility and keeps the course CSS small.
