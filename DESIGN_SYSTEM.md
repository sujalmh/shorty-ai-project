# Design System Documentation

## Overview
This document outlines the comprehensive design system redesign implemented for the AI Excel spreadsheet application, creating a professional, modern, and cohesive user interface.

## Design Tokens

### Color Palette
- **Primary**: Indigo (600-700) - Main interactive elements
- **Secondary**: Slate (600-700) - Supporting elements
- **Accent**: Blue (500-600) - Highlights and special cases
- **Success**: Emerald (500-600) - Positive actions
- **Warning**: Amber (500-600) - Warnings
- **Error**: Red (500-600) - Errors
- **Neutral**: Gray (100-900) - Text and backgrounds

### Spacing System
- Container padding: `1rem` (16px)
- Component padding: `0.75rem` (12px)
- Element padding: `0.5rem` (8px)
- Consistent gaps: `0.5rem`, `0.75rem`, `1rem`

### Typography
- Font sizes: 12px (xs), 14px (sm), 16px (base), 18px (lg), 20px (xl), 24px (2xl)
- Font weights: normal (400), medium (500), semibold (600), bold (700)
- Line heights: Optimized for readability

### Border Radius
- Small: `0.375rem` (6px)
- Medium: `0.5rem` (8px)
- Large: `0.75rem` (12px)
- XL: `1rem` (16px)
- 2XL: `1.5rem` (24px) - Main glass containers

## Component Styles

### Buttons
All buttons now follow a standardized design:

**Dimensions:**
- Height: `h-9` (36px)
- Padding: `px-4 py-2`
- Border radius: `rounded-lg`

**Variants:**
1. **Primary** (`btn-primary`):
   - Background: indigo-600
   - Text: white
   - Hover: indigo-700
   - Use for: Main actions, submit buttons

2. **Secondary** (`btn-secondary`):
   - Background: white
   - Border: gray-300
   - Text: gray-700
   - Hover: gray-50
   - Use for: Secondary actions, cancel buttons

3. **Tertiary** (`btn-tertiary`):
   - Background: transparent
   - Border: gray-300
   - Hover: white/50
   - Use for: Subtle actions

4. **Ghost** (`btn-ghost`):
   - Background: white/10
   - Border: white/20
   - Text: white
   - Use for: Actions on dark backgrounds (chat panel)

5. **Danger** (`btn-danger`):
   - Background: red-600
   - Text: white
   - Use for: Destructive actions

**Features:**
- Active state: `scale-[0.98]` for tactile feedback
- Focus ring: `ring-2` with appropriate color
- Disabled state: `opacity-50` + `cursor-not-allowed`
- Smooth transitions: `transition-all duration-200`

### Input Fields
Standardized input styling:

**Dimensions:**
- Height: `h-9` (36px)
- Padding: `px-3 py-2`
- Border radius: `rounded-lg`

**Variants:**
1. **Light** (`input-light`):
   - Background: white
   - Border: gray-300
   - Use for: Light backgrounds (sheet area)

2. **Dark** (`input-dark`):
   - Background: white/10
   - Border: white/20
   - Text: white
   - Use for: Dark backgrounds (chat panel)

**Features:**
- Focus state: `ring-2 ring-indigo-500`
- Placeholder: Subtle text color
- Smooth transitions

### Tabs
**Dimensions:**
- Height: `h-9` (36px)
- Padding: `px-4 py-2`
- Border radius: `rounded-lg`

**States:**
- Active: `tab-btn-active` (indigo-600 background, white text, shadow)
- Inactive: `tab-btn-inactive` (white/80 background, gray-700 text, border)

**Features:**
- Badge support for counts
- Active scale effect
- Focus rings

### Message Bubbles (Chat)
**User Messages:**
- Background: indigo-600
- Text: white
- Border radius: `rounded-xl`
- Padding: `px-4 py-2`
- Alignment: right

**Assistant Messages:**
- Background: white/10
- Border: white/20
- Text: white
- Border radius: `rounded-xl`
- Padding: `px-4 py-2`
- Alignment: left

## Glass Morphism Effects

### Glass Dark (Chat Panel)
```css
background: gradient from slate-900/40 to slate-800/30
backdrop-blur: xl
border: white/10
border-radius: 2xl
shadow: [0_8px_32px_rgba(0,0,0,0.4)]
```

### Glass Light (Sheet/Diagrams Panel)
```css
background: white/80
backdrop-blur: xl
border: gray-200/50
border-radius: 2xl
shadow: [0_8px_32px_rgba(0,0,0,0.08)]
```

### Header Glass
```css
background: white/70
backdrop-blur: md
border-bottom: gray-200/60
```

## AG Grid Integration

Custom theming to match the design system:

**Variables:**
- Background: transparent
- Odd rows: slate-50/50 (subtle alternation)
- Header background: white/70
- Header text: slate-700
- Border color: slate-200
- Row hover: indigo/5
- Selected row: indigo/10
- Font size: 14px
- Row height: 40px
- Header height: 40px

**Highlighted Rows:**
- Background: indigo-50/80
- Left border: 4px solid indigo-500
- Dark mode: indigo-900/40

**Features:**
- Rounded container with border
- Medium font weight headers
- Cell borders for clear separation
- Smooth hover transitions

## Scrollbars

Custom styled scrollbars for better aesthetics:

**Dimensions:**
- Width/Height: 8px

**Styling:**
- Track: transparent
- Thumb: gray-300 (light) / gray-600 (dark)
- Thumb hover: gray-400 (light) / gray-500 (dark)
- Border radius: full (pill-shaped)

## Layout Improvements

### App Layout
- Consistent `gap-4` between panels
- Outer padding: `p-4`
- Grid system: 1 column (mobile), 3 columns (desktop)
- Chat panel: 1 column
- Sheet/Diagrams: 2 columns

### Component Spacing
- Component padding: `component-padding` class (p-3)
- Section spacing: `section-spacing` class (space-y-3)
- Inline spacing: `inline-spacing` class (gap-2)

### Sheet Toolbar
- Two logical rows with clear separation
- First row: Data operations (Import, Add Row, Add Column)
- Second row: Computed column creation
- Visual dividers between sections
- Consistent button spacing and alignment

## Accessibility

### Contrast Ratios
All text meets WCAG AA standards:
- White on indigo-600: 4.5:1+
- Gray-700 on white: 7:1+
- White on slate-900: 15:1+

### Focus States
- Visible focus rings on all interactive elements
- Color: indigo-500 (2px width)
- High contrast for keyboard navigation

### ARIA Labels
- Maintained existing ARIA labels
- Screen reader support for chat messages
- Semantic HTML structure

## Component-Specific Improvements

### Chat Panel
- ✅ Standardized message bubble styling
- ✅ Better input/button alignment
- ✅ Consistent quick prompt buttons
- ✅ Improved mode selector styling
- ✅ Custom scrollbar

### Sheet Grid
- ✅ Reorganized toolbar with clear sections
- ✅ Consistent button heights and styles
- ✅ Improved input field alignment
- ✅ Visual dividers between sections
- ✅ Better label typography
- ✅ AG Grid theming integration

### Diagrams
- ✅ Cleaner empty state message
- ✅ Better spacing between plots
- ✅ Custom scrollbar

### Plot Viewer
- ✅ Consistent glass effect styling
- ✅ Standardized header with semibold titles
- ✅ Improved close button styling
- ✅ Better padding and spacing

### App Header/Tabs
- ✅ Badge component for diagram count
- ✅ Clear active/inactive tab states
- ✅ Consistent button styling for "Clear All"

## Best Practices Implemented

1. **Consistency**: All similar elements use the same styles
2. **Visual Hierarchy**: Clear distinction between primary, secondary, and tertiary actions
3. **Spacing**: Predictable and consistent spacing throughout
4. **Color Usage**: Meaningful color choices with semantic purpose
5. **Transitions**: Smooth, subtle animations for better UX
6. **Accessibility**: WCAG compliant contrast and focus states
7. **Scalability**: Utility classes for easy maintenance
8. **Responsiveness**: Flexible layouts that work on different screen sizes

## Future Enhancements

Potential improvements for future iterations:
- Dark mode toggle with refined dark theme
- Additional color themes
- Animation preferences (respect prefers-reduced-motion)
- More semantic color usage (info, success toasts)
- Enhanced mobile responsiveness
- Keyboard shortcuts overlay

## Migration Notes

The design system uses:
- Tailwind CSS utilities
- Custom CSS classes in `index.css`
- CSS variables for easy theming
- No breaking changes to functionality
- Backward compatible with existing components
