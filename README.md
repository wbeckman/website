# Will Beckman Blog

A Jekyll-based personal blog hosted on GitHub Pages.

## Quick Start

### Local Development

```bash
# Install dependencies (first time only)
bundle install

# Run locally
bundle exec jekyll serve

# Site will be available at http://localhost:4000
```

### Adding a New Blog Post

1. Create a new file in `_posts/` with the naming convention:
   ```
   YYYY-MM-DD-your-post-title.markdown
   ```

2. Add front matter at the top:
   ```yaml
   ---
   layout: post
   title: "Your Post Title"
   date: YYYY-MM-DD HH:MM:SS -0500
   usemathjax: true  # optional, for LaTeX rendering
   ---
   ```

3. Write your content in Markdown below the front matter.

### Adding Images to Posts

1. Put images in `assets/img/` or `assets/posts/your-post-name/`
2. Reference in your post:
   ```markdown
   ![Alt text](/assets/img/your-image.png)
   ```

## Deployment

Push to `main` and GitHub Actions will build and deploy automatically. Track progress under the **Actions** tab in the repo. Changes typically appear within 1-2 minutes.

## Project Structure

```
.
├── _config.yml      # Site configuration
├── _includes/       # Reusable HTML partials
├── _layouts/        # Page templates
├── _posts/          # Blog posts (YYYY-MM-DD-title.markdown)
├── assets/          # Static files (images, CSS, PDFs)
│   ├── img/
│   └── posts/
├── index.md         # Homepage
├── about.md         # About page
└── 404.html         # Error page
```

## Common Tasks

### Editing Site Metadata

Edit `_config.yml` to change:
- `title` - Site title
- `description` - Site description
- `email` - Contact email
- `github_username` - GitHub profile link

**Note:** After editing `_config.yml`, restart the local server to see changes.

### Embedding JS Animations

Jekyll passes raw HTML through unchanged, so you can drop JavaScript directly into any `.markdown` post.

**Inline canvas animation**

```html
<canvas id="my-canvas" width="600" height="300" style="border:1px solid #ccc;"></canvas>
<script>
  const canvas = document.getElementById('my-canvas');
  const ctx = canvas.getContext('2d');
  let x = 0;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillRect(x % canvas.width, 130, 20, 20);
    x += 2;
    requestAnimationFrame(draw);
  }
  draw();
</script>
```

**Load a library from CDN, then use it**

```html
<div id="plot"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  Plotly.newPlot('plot', [{
    x: [1, 2, 3],
    y: [2, 6, 3],
    type: 'scatter'
  }]);
</script>
```

**Load a local JS file from `assets/`**

Put your script at `assets/js/my-animation.js`, then in the post:

```html
<div id="root"></div>
<script src="/assets/js/my-animation.js"></script>
```

**Tip:** If the script depends on the DOM element being present, place the `<script>` tag *after* the element, not in `<head>`.

---

### Using Math (LaTeX)

Add `usemathjax: true` to your post's front matter, then use:
- Inline: `$E = mc^2$`
- Block: `$$\sum_{i=1}^{n} x_i$$`

### Drafts

Create posts in a `_drafts/` folder (without dates in filename). Preview with:
```bash
bundle exec jekyll serve --drafts
```

### Updating Dependencies

```bash
bundle update
```

## Theme

This site uses the [Minima](https://github.com/jekyll/minima) theme. To customize:
1. Create `_sass/` folder for custom styles
2. Override layouts by copying them from the theme to `_layouts/`

## Troubleshooting

**Site not updating after push?**
- Check GitHub Actions for build errors
- Clear browser cache

**Local server errors?**
```bash
bundle install  # Reinstall dependencies
```

**Ruby version issues?**
Use a version manager like rbenv or asdf. The `github-pages` gem pins its own Ruby/Jekyll versions — match those locally for the closest parity.
