-- Render a locally exported marimo WASM notebook in HTML and preserve the
-- fenced Div contents as the fallback for non-HTML formats.
--
-- Usage:
-- ::: {.quarto-wasm-local notebook="example.edit.py" height="720"}
-- ::: {.quarto-wasm-local notebook="dashboard.py" fullscreen="true"}
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
  local editable = notebook:match("%.edit%.py$") ~= nil
  local direct_src = src
  if editable then
    -- Keep the embedded view compact. The direct link uses direct_src without
    -- this query parameter, so marimo opens with its full editing chrome.
    src = src .. "?show-chrome=false"
  end

  local title = escape_html(div.attributes.title or "Locally hosted marimo notebook")
  local height = escape_html(div.attributes.height or "700")
  local loading = escape_html(div.attributes.loading or "lazy")
  if loading ~= "lazy" and loading ~= "eager" then
    error("A .quarto-wasm-local loading attribute must be lazy or eager")
  end

  local fullscreen = div.attributes.fullscreen or "false"
  if fullscreen ~= "true" and fullscreen ~= "false" then
    error("A .quarto-wasm-local fullscreen attribute must be true or false")
  end

  local controls = {}
  if editable then
    table.insert(
      controls,
      '<button class="quarto-wasm-view-toggle" type="button" disabled>App view</button>'
    )
  end
  if fullscreen == "true" then
    table.insert(
      controls,
      '<button class="quarto-wasm-fullscreen" type="button">Full screen</button>'
    )
  end

  local controls_html = ""
  if #controls > 0 then
    controls_html = '<div class="quarto-wasm-controls">'
      .. table.concat(controls, "\n")
      .. "</div>"
  end

  local hint_html = ""
  if editable then
    hint_html = '<p class="marimo-toggle-hint">Use the <strong>App view / Edit code</strong> button, or click inside the notebook and press <kbd>⌘.</kbd> on macOS or <kbd>Ctrl+.</kbd> elsewhere.</p>'
  end

  src = escape_html(src)
  direct_src = escape_html(direct_src)
  local iframe = string.format(
    [[<div class="marimo-embed-frame quarto-wasm-local-frame">
  %s
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
  %s
  <details class="marimo-troubleshooting">
    <summary>Open or reset notebook</summary>
    <p>The first start may take up to a minute while the browser downloads Python and the required packages.</p>
    <div class="marimo-troubleshooting-actions">
      <a href="%s" target="_blank" rel="noopener">Open notebook directly</a>
      <button class="quarto-wasm-reload" type="button">Reload/reset notebook</button>
    </div>
    <p class="marimo-troubleshooting-note">Reloading resets unsaved changes. If the direct page also fails, send the course page URL, browser name, and a screenshot to the teaching team.</p>
  </details>
</div>
<script>
(() => {
  const script = document.currentScript;
  const wrapper = script.previousElementSibling;
  const viewButton = wrapper.querySelector(".quarto-wasm-view-toggle");
  const fullscreenButton = wrapper.querySelector(".quarto-wasm-fullscreen");
  const reloadButton = wrapper.querySelector(".quarto-wasm-reload");
  const iframe = wrapper.querySelector("iframe");

  if (viewButton) {
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
      viewButton.textContent = editing ? "App view" : "Edit code";
      viewButton.setAttribute("aria-pressed", editing ? "false" : "true");
    };

    const ready = () => {
      viewButton.disabled = false;
    };

    iframe.addEventListener("load", ready);
    if (iframe.contentDocument && iframe.contentDocument.readyState === "complete") {
      ready();
    }

    viewButton.addEventListener("click", () => {
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
  }

  if (reloadButton) {
    reloadButton.addEventListener("click", () => {
      iframe.src = iframe.src;
    });
  }

  if (fullscreenButton) {
    const requestFullscreen =
      iframe.requestFullscreen || iframe.webkitRequestFullscreen;
    if (!requestFullscreen) {
      fullscreenButton.hidden = true;
    } else {
      fullscreenButton.addEventListener("click", async () => {
        try {
          await requestFullscreen.call(iframe);
        } catch (_) {
          // The browser or an outer iframe may deny fullscreen.
        }
      });
    }
  }
})();
</script>]],
    controls_html,
    src,
    title,
    height,
    loading,
    hint_html,
    direct_src
  )

  return pandoc.RawBlock("html", iframe)
end
