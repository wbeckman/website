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
   categories: [category1, category2]
   usemathjax: true  # optional, if you need math rendering
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

This site auto-deploys via GitHub Pages when you push to `main`.

```bash
git add .
git commit -m "Your commit message"
git push origin main
```

The site will rebuild automatically. Changes typically appear within 1-2 minutes.

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
Check `.ruby-version` or use a version manager like rbenv/asdf.
