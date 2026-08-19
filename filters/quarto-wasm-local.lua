-- Render a locally exported marimo WASM notebook in HTML and preserve the
-- fenced Div contents as the fallback for non-HTML formats.
--
-- Usage:
-- ::: {.quarto-wasm-local notebook="example.edit.py" height="720"}
-- Static/PDF fallback.
-- :::

local function escape_html(value)
  return (value
    :gsub("&", "&amp;")
    :gsub('"', "&quot;")
    :gsub("<", "&lt;")
    :gsub(">", "&gt;"))
end

local function exported_stem(filename)
  local stem = filename:gsub("%.py$", "")
  return stem:gsub("%.edit$", "")
end

function Div(div)
  if not div.classes:includes("quarto-wasm-local") then
    return nil
  end

  if not quarto.doc.is_format("html") then
    return div.content
  end

  local notebook = div.attributes.notebook or div.attributes.source
  if not notebook or notebook == "" then
    error("A .quarto-wasm-local Div requires a notebook attribute")
  end
  if notebook:find("[/\\]") then
    error(
      ".quarto-wasm-local notebook must be a filename relative to activities/: "
        .. notebook
    )
  end
  if not notebook:match("%.py$") then
    error("A .quarto-wasm-local notebook must end in .py: " .. notebook)
  end

  local project_dir = quarto.project.directory or "."
  local source_path = pandoc.path.join({ project_dir, "activities", notebook })
  local handle = io.open(source_path, "rb")
  if not handle then
    error("Cannot read local WASM notebook: " .. source_path)
  end
  handle:close()

  local offset = quarto.project.offset or "."
  local output_name = exported_stem(notebook) .. ".html"
  local src = pandoc.path.join({ offset, "wasm-local", output_name })
  if notebook:match("%.edit%.py$") then
    src = src .. "?show-chrome=false"
  end

  local title = escape_html(div.attributes.title or "Locally hosted marimo notebook")
  local height = escape_html(div.attributes.height or "700")
  local loading = escape_html(div.attributes.loading or "lazy")
  if loading ~= "lazy" and loading ~= "eager" then
    error("A .quarto-wasm-local loading attribute must be lazy or eager")
  end
  src = escape_html(src)

  local iframe = string.format(
    [[<div class="marimo-embed-frame quarto-wasm-local-frame">
  <button class="quarto-wasm-view-toggle" type="button" disabled>App view</button>
  <iframe
    src="%s"
    title="%s"
    width="100%%"
    height="%s"
    style="border: 1px solid #dee2e6; border-radius: 0.5rem;"
    sandbox="allow-scripts allow-same-origin allow-downloads allow-popups allow-forms"
    allow="clipboard-read; clipboard-write; microphone"
    allowfullscreen
    loading="%s"
  ></iframe>
  <p class="marimo-toggle-hint">Use the <strong>App view / Edit code</strong> button, or click inside the notebook and press <kbd>⌘.</kbd> on macOS or <kbd>Ctrl+.</kbd> elsewhere.</p>
</div>
<script>
(() => {
  const script = document.currentScript;
  const wrapper = script.previousElementSibling;
  const button = wrapper.querySelector(".quarto-wasm-view-toggle");
  const iframe = wrapper.querySelector("iframe");

  const editorsAreVisible = () => {
    try {
      return Array.from(iframe.contentDocument.querySelectorAll(".cm-editor"))
        .some((editor) => editor.getBoundingClientRect().height > 0);
    } catch (_) {
      return false;
    }
  };

  const updateLabel = () => {
    const editing = editorsAreVisible();
    button.textContent = editing ? "App view" : "Edit code";
    button.setAttribute("aria-pressed", editing ? "false" : "true");
  };

  const ready = () => {
    // Local editable exports start in notebook view, matching the label
    // authored above. Subsequent clicks update the label from the DOM state.
    button.disabled = false;
  };

  iframe.addEventListener("load", ready);
  if (iframe.contentDocument && iframe.contentDocument.readyState === "complete") {
    ready();
  }

  button.addEventListener("click", () => {
    const childWindow = iframe.contentWindow;
    const childDocument = iframe.contentDocument;
    if (!childWindow || !childDocument) return;

    const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
    for (const type of ["keydown", "keyup"]) {
      childDocument.dispatchEvent(new childWindow.KeyboardEvent(type, {
        key: ".",
        code: "Period",
        metaKey: isMac,
        ctrlKey: !isMac,
        bubbles: true,
      }));
    }
    window.setTimeout(updateLabel, 150);
  });
})();
</script>]],
    src,
    title,
    height,
    loading
  )

  return pandoc.RawBlock("html", iframe)
end
