// Icons — minimal line icons
const Icon = ({ name, size = 16, stroke = 1.6, style = {} }) => {
  const s = size, sw = stroke;
  const props = { width: s, height: s, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: sw, strokeLinecap: 'round', strokeLinejoin: 'round', style };
  switch (name) {
    case 'play':    return <svg {...props}><path d="M7 5 L19 12 L7 19 Z" fill="currentColor" stroke="none"/></svg>;
    case 'pause':   return <svg {...props}><rect x="7" y="5" width="3.5" height="14" fill="currentColor" stroke="none"/><rect x="13.5" y="5" width="3.5" height="14" fill="currentColor" stroke="none"/></svg>;
    case 'back':    return <svg {...props}><path d="M19 5 L9 12 L19 19 Z" fill="currentColor" stroke="none"/><rect x="5" y="5" width="2" height="14" fill="currentColor" stroke="none"/></svg>;
    case 'forward': return <svg {...props}><path d="M5 5 L15 12 L5 19 Z" fill="currentColor" stroke="none"/><rect x="17" y="5" width="2" height="14" fill="currentColor" stroke="none"/></svg>;
    case 'volume':  return <svg {...props}><path d="M4 9 L4 15 L8 15 L13 19 L13 5 L8 9 Z"/><path d="M16 8 Q18 12 16 16"/><path d="M18 6 Q21 12 18 18"/></svg>;
    case 'caption': return <svg {...props}><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 12h3M7 15h2M13 12h4M13 15h3"/></svg>;
    case 'full':    return <svg {...props}><path d="M4 9 V4 H9 M20 9 V4 H15 M4 15 V20 H9 M20 15 V20 H15"/></svg>;
    case 'refresh': return <svg {...props}><path d="M3 12a9 9 0 1 1 3.5 7.1"/><path d="M3 20v-5h5"/></svg>;
    case 'edit':    return <svg {...props}><path d="M4 20h4l10-10-4-4-10 10z"/><path d="M14 6l4 4"/></svg>;
    case 'play-s':  return <svg {...props}><path d="M8 5 L19 12 L8 19 Z" fill="currentColor" stroke="none"/></svg>;
    case 'check':   return <svg {...props}><path d="M5 12l5 5L20 7"/></svg>;
    case 'x':       return <svg {...props}><path d="M6 6 L18 18 M6 18 L18 6"/></svg>;
    case 'chevr':   return <svg {...props}><path d="M9 6l6 6-6 6"/></svg>;
    case 'chevd':   return <svg {...props}><path d="M6 9l6 6 6-6"/></svg>;
    case 'chevu':   return <svg {...props}><path d="M18 15l-6-6-6 6"/></svg>;
    case 'arrow':   return <svg {...props}><path d="M5 12h14M13 6l6 6-6 6"/></svg>;
    case 'file':    return <svg {...props}><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/></svg>;
    case 'film':    return <svg {...props}><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M3 15h18M8 4v16M16 4v16"/></svg>;
    case 'upload':  return <svg {...props}><path d="M12 16V4M6 10l6-6 6 6"/><path d="M4 18v2h16v-2"/></svg>;
    case 'mic':     return <svg {...props}><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8"/></svg>;
    case 'wave':    return <svg {...props}><path d="M3 12h2M7 8v8M11 5v14M15 9v6M19 11v2M21 12h-0"/></svg>;
    case 'sparkle': return <svg {...props}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6"/></svg>;
    case 'cog':     return <svg {...props}><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2"/></svg>;
    case 'trash':   return <svg {...props}><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></svg>;
    case 'save':    return <svg {...props}><path d="M5 3h11l4 4v14H5z"/><path d="M8 3v6h8V3M8 14h8v7H8z"/></svg>;
    case 'logs':    return <svg {...props}><path d="M4 6h16M4 12h16M4 18h10"/></svg>;
    case 'info':    return <svg {...props}><circle cx="12" cy="12" r="9"/><path d="M12 8v.01M11 12h1v5h1"/></svg>;
    case 'plus':    return <svg {...props}><path d="M12 5v14M5 12h14"/></svg>;
    default: return null;
  }
};

window.Icon = Icon;
