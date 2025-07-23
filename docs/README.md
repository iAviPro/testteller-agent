# TestTeller Documentation Site

This folder contains the Jekyll-powered documentation site for TestTeller.

## Local Development

### Prerequisites

- Ruby (version 2.5.0 or higher)
- Bundler (`gem install bundler`)

### Setup

1. Navigate to the docs directory:
   ```bash
   cd docs/
   ```

2. Install dependencies:
   ```bash
   bundle install
   ```

3. Serve the site locally:
   ```bash
   bundle exec jekyll serve
   ```

4. Open your browser to `http://localhost:4000/testteller-agent/`

### Live Reload

For development with live reload:
```bash
bundle exec jekyll serve --livereload
```

## GitHub Pages Deployment

This site is automatically deployed to GitHub Pages when changes are pushed to the main branch. The site will be available at:

```
https://iAviPro.github.io/testteller-agent/
```

## Theme Customization

The site uses a custom minimal theme. To customize:

1. **Colors and Typography**: Edit variables in `assets/css/style.css`
2. **Layout**: Modify `_layouts/default.html`
3. **Navigation**: Update the `navigation` section in `_config.yml`

### Available Theme Options

You can also use GitHub Pages supported themes by updating `_config.yml`:

```yaml
# Minimal theme (clean and simple)
remote_theme: pages-themes/minimal@v0.2.0

# Cayman theme (modern with gradient header)
remote_theme: pages-themes/cayman@v0.2.0

# Slate theme (dark and professional)
remote_theme: pages-themes/slate@v0.2.0

# Architect theme (clean with blueprints aesthetic)
remote_theme: pages-themes/architect@v0.2.0

# Hacker theme (terminal-style dark theme)
remote_theme: pages-themes/hacker@v0.2.0
```

## File Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── _layouts/            # Page layouts
│   └── default.html     # Main layout template
├── assets/              # Static assets
│   └── css/
│       ├── style.css    # Main stylesheet
│       └── syntax.css   # Code syntax highlighting
├── index.md             # Homepage
├── ARCHITECTURE.md      # Architecture documentation
├── COMMANDS.md          # CLI commands reference
├── FEATURES.md          # Features documentation
├── TESTING.md           # Testing guide
├── Gemfile              # Ruby dependencies
└── README.md            # This file
```

## Writing Documentation

### Front Matter

All markdown files should include front matter:

```yaml
---
layout: default
title: Page Title
description: Page description for SEO
---
```

### Syntax Highlighting

Use fenced code blocks with language specification:

````markdown
```python
def hello_world():
    print("Hello, World!")
```
````

### Linking

- Internal links: `[Architecture](ARCHITECTURE.md)`
- External links: `[GitHub](https://github.com/iAviPro/testteller-agent)`
- Anchor links: `[Section](#section-id)`

## Troubleshooting

### Bundle Install Fails

If you encounter issues with `bundle install`:

```bash
# Update bundler
gem update bundler

# Try installing with specific platform
bundle lock --add-platform x86_64-linux
bundle install
```

### Jekyll Serve Fails

If Jekyll won't start:

```bash
# Clean Jekyll cache
bundle exec jekyll clean

# Rebuild site
bundle exec jekyll build

# Then serve
bundle exec jekyll serve
```

### Page Not Found

Make sure your baseurl is set correctly in `_config.yml`:

```yaml
baseurl: "/testteller-agent"  # For GitHub Pages
# or
baseurl: ""  # For custom domain
```

## Contributing

When adding new documentation:

1. Create a new `.md` file in the docs folder
2. Add appropriate front matter
3. Update navigation in `_config.yml` if needed
4. Test locally before pushing