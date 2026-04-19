// Auto Dub — shared UI data.
// Segments and logs come from the Python bridge at runtime; only the
// static language list lives here.

const LANGUAGES = [
  { code: 'auto',       label: 'Auto-detect', short: 'A'  },
  { code: 'chinese',    label: 'Chinese',     short: '中' },
  { code: 'english',    label: 'English',     short: 'En' },
  { code: 'japanese',   label: 'Japanese',    short: '日' },
  { code: 'korean',     label: 'Korean',      short: '한' },
  { code: 'spanish',    label: 'Spanish',     short: 'Es' },
  { code: 'french',     label: 'French',      short: 'Fr' },
  { code: 'german',     label: 'German',      short: 'De' },
  { code: 'italian',    label: 'Italian',     short: 'It' },
  { code: 'portuguese', label: 'Portuguese',  short: 'Pt' },
  { code: 'russian',    label: 'Russian',     short: 'Ru' },
];

window.LANGUAGES = LANGUAGES;
