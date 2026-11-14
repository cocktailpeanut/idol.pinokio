---
name: IDOL Canvas
description: A quiet light canvas for turning one image into a 3D result.
colors:
  white: "#fff"
  ink: "#26382f"
  muted: "#657268"
  accent: "#345646"
  accent-hover: "#294638"
  model-tray: "#f2f5f1"
  model-unselected: "#59665d"
  selection-fill: "#dce8dd"
  activity-track: "#e5ebe5"
  error-ink: "#743a2d"
  error-border: "#e8d9d4"
  error-surface: "#fffaf7"
typography:
  title:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "24px"
    fontWeight: 400
    letterSpacing: "-.025em"
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "14px"
  label:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "14px"
    lineHeight: "20px"
  label-compact:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "13px"
    lineHeight: "20px"
  helper:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "12px"
rounded:
  primary: "6px"
  segment: "5px"
  segment-group: "8px"
  feedback: "10px"
  drop-frame: "12px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.white}"
    typography: "{typography.label}"
    rounded: "{rounded.primary}"
    padding: "11px 20px"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
  button-quiet:
    backgroundColor: "{colors.white}"
    textColor: "{colors.muted}"
    typography: "{typography.label}"
    padding: "10px 0"
  model-segment:
    textColor: "{colors.model-unselected}"
    typography: "{typography.label}"
    rounded: "{rounded.segment}"
    padding: "7px 13px"
  model-segment-selected:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
---

# Design System: IDOL Canvas

## Overview

**Creative North Star: "Canvas"**

The page is the tool: a white field gives the uploaded image and generated object the entire visual stage. System sans text, muted green controls, and generous empty space keep the interface quiet. Controls appear only when useful.

**Key Characteristics:**
- One full-viewport working surface.
- A small type hierarchy with no display typography.
- Muted green action emphasis and neutral supporting controls.
- Flat surfaces with one subtle selected-control shadow.

## Colors

The palette is white and green-tinted neutrals, with a darker green reserved for action and focus.

### Primary
- **Muted Forest:** Generate, Download, keyboard focus, and the activity indicator; its deeper hover variant signals interaction.

### Neutral
- **White:** page, viewer clear color, primary-button text, and selected model segment.
- **Green Ink:** primary text and the selected model label.
- **Muted Sage:** helper text, status copy, and New image.
- **Pale Sage:** the model selector tray; a quieter gray-green distinguishes unselected labels.
- **Selection Sage:** the background behind selected text, paired with Green Ink.
- **Activity Track:** the quiet track behind the moving progress stroke.
- **Clay:** error text and setup links, paired with the documented warm Error Surface and Error Border.

**The Action Emphasis Rule.** Use the darkest filled green control for the next main action. Supporting controls remain quiet.

## Typography

One native system sans stack serves all interface text. The largest text is the regular-weight upload instruction; this is a tool prompt, not a marketing headline.

- **Title:** upload and drag-over prompts; the tight tracking keeps short text compact.
- **Body:** upload hint, activity, and errors.
- **Label:** model choice and actions, with the established control line height.
- **Compact Label:** the narrow-screen action and model-control text.
- **Helper:** the result interaction hint.

Control labels reduce to (13px) below the narrow-screen breakpoint. There is no secondary display or monospace family.

## Layout

The canvas occupies the dynamic viewport with a minimum height of (340px). The empty prompt is centered on both axes. Images remain centered and contained, leaving (80px) total horizontal clearance and (190px) vertical clearance on desktop. The 3D viewer fills the canvas.

Controls form a single centered bottom dock with a (22px) gap and a (28px) minimum bottom inset, respecting the device safe area. At widths up to (540px), its gap and bottom inset become (14px) and (22px), image clearance becomes (32px) horizontally and (170px) vertically, and control padding contracts. Coarse-pointer actions have a minimum height of (44px).

**The Content Stage Rule.** Keep the main artifact on the canvas and supporting controls at its edge. The Canvas surface has no navbar, sidebar, or permanent settings panel.

## Elevation & Depth

The interface is flat. White space, tonal contrast, and placement separate content and controls; there are no decorative cards or dock backplates. Only the selected model segment has a faint shadow, recorded in the sidecar. The generated 3D object supplies the scene's visual depth.

## Shapes

Controls use small, gently rounded corners. The model group encloses slightly tighter segments; primary actions have their own modest radius. Drag feedback alone creates a near-edge dashed frame using the Drop Frame radius. Error feedback uses the Feedback radius on its small warm bordered container. No ornamental shape system is present.

## Components

### Actions

Generate and Download share the filled primary style. New image is a borderless text action; hover darkens its label. Keyboard focus is a visible green outline with an offset, inset for the full-page upload target and viewer. Controls change color immediately; there is no decorative transition.

### Model selector

Turbo and IDOL form one compact segmented group. A white selected segment, darker label, and faint shadow carry selection together with `aria-pressed`. The group is shown only while an image is ready to generate.

### Canvas states

The empty state contains only the centered upload prompt and hint. The image state adds the bottom dock. Generation fades and raises the image to leave space for the current stage, sampling step count, elapsed time, and a thin activity line. Elapsed time uses the existing Helper typography and tabular numerals. The result uses the full-page viewer with Download, New image, and a small orbit hint. The exact surface sequence lives in the Canvas surface brief.

### Feedback

Dragging a file produces a translucent pale-green overlay and dashed border. Errors appear above the dock; setup links appear only when applicable. Preview failure retains the Download action. The activity line stops moving under reduced-motion preferences.

## Do's and Don'ts

### Do:
- **Do** keep the artifact visually dominant and controls compact.
- **Do** use the existing green action, focus, and selection language.
- **Do** retain visible keyboard focus, safe-area spacing, and coarse-pointer target sizes.
- **Do** make progress and failure feedback truthful and keep completed files downloadable.

### Don't:
- **Don't** add navigation or permanent settings panels to the Canvas surface.
- **Don't** add display typography, decorative cards, gradients, or ornamental motion to this quiet tool.
- **Don't** turn New image into a destructive action on saved results.
