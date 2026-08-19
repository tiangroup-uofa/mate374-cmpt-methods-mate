-- Turn a fenced Div into an editable molab iframe sourced from a local
-- marimo notebook. Source is compressed into molab's `#code/` URL, so no
-- standalone HTML export is needed.

local function read_file(path)
  local handle, message = io.open(path, "rb")
  if not handle then
    error("Cannot read marimo iframe source " .. path .. ": " .. tostring(message))
  end
  local contents = handle:read("*all")
  handle:close()
  return contents
end

local function escape_html(value)
  return (value
    :gsub("&", "&amp;")
    :gsub('"', "&quot;")
    :gsub("<", "&lt;")
    :gsub(">", "&gt;"))
end

function Div(div)
  if not div.classes:includes("marimo-iframe") then
    return nil
  end

  -- Preserve the Div's contents as the printable/static fallback.
  if not quarto.doc.is_format("html") then
    return div.content
  end

  local source = div.attributes.source
  if not source or source == "" then
    error("A .marimo-iframe Div requires a source attribute")
  end

  local project_dir = quarto.project.directory or "."
  local source_path = pandoc.path.join({ project_dir, source })
  read_file(source_path) -- Fail early with a clear source-path error.

  local helper = pandoc.path.join({ project_dir, "scripts", "marimo_iframe_url.py" })
  local raw_src = pandoc.pipe(
    "uv",
    { "run", "--quiet", "--project", project_dir, helper, source_path },
    ""
  )
  raw_src = raw_src:gsub("%s+$", "")

  local title = escape_html(div.attributes.title or "Editable marimo notebook")
  local height = escape_html(div.attributes.height or "700")
  local src = escape_html(raw_src)

  local iframe = string.format(
    [[<div class="marimo-embed-frame">
  <iframe
    src="%s"
    title="%s"
    width="100%%"
    height="%s"
    style="border: 1px solid #dee2e6; border-radius: 0.5rem;"
    sandbox="allow-scripts allow-same-origin allow-downloads allow-popups allow-forms"
    allow="clipboard-read; clipboard-write; microphone"
    allowfullscreen
    loading="lazy"
  ></iframe>
  <p class="marimo-toggle-hint">Click inside the notebook, then press  <kbd>Ctrl+.</kbd> (Windows/Linux) <kbd>⌘.</kbd> (macOS) to switch between app and edit views.</p>
</div>]],
    src,
    title,
    height
  )

  return pandoc.RawBlock("html", iframe)
end
