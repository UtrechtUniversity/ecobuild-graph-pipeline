import React, { useState } from 'react';

export interface SearchableSelectOption {
  value: string;
  label: string;
}

const SearchableSelect: React.FC<{
  options: SearchableSelectOption[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}> = ({ options, value, onChange, placeholder }) => {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);

  const selectedLabel = options.find((o) => o.value === value)?.label ?? '';
  const filtered = options.filter((o) => o.label.toLowerCase().includes(query.toLowerCase()));

  return (
    <div className="relative">
      <input
        type="text"
        value={open ? query : selectedLabel}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => { setQuery(''); setOpen(true); }}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => { if (e.key === 'Escape') e.currentTarget.blur(); }}
        placeholder={placeholder}
        className="h-9 w-full rounded-md border border-input bg-card px-3 text-sm shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
      />
      {open && filtered.length > 0 && (
        // mousedown fires (and can select) before the input's onBlur closes this list
        <ul onMouseDown={(e) => e.preventDefault()} className="absolute z-10 mt-1 max-h-60 w-full overflow-y-auto rounded-md border border-input bg-card shadow-md">
          {filtered.map((o) => (
            <li
              key={o.value}
              onClick={() => { onChange(o.value); setOpen(false); }}
              className="cursor-pointer px-3 py-1.5 text-sm hover:bg-accent"
            >
              {o.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SearchableSelect;
