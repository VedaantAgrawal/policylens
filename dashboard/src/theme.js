import { useEffect, useState } from 'react'

// Validated categorical palette (see the dataviz work in this repo's history)
// — slots 1-3 pass all-pairs CVD/contrast validation in both modes; slot 4
// (yellow) is used here only because every bar in this dashboard also carries
// a text label (the stage name), so color is never the sole identity channel.
export const PALETTE = {
  light: {
    series1: '#2a78d6',
    series2: '#eb6834',
    series3: '#1baf7a',
    series4: '#eda100',
    surface: '#fcfcfb',
    page: '#f9f9f7',
    textPrimary: '#0b0b0b',
    textSecondary: '#52514e',
    gridline: '#e1e0d9',
  },
  dark: {
    series1: '#3987e5',
    series2: '#d95926',
    series3: '#199e70',
    series4: '#c98500',
    surface: '#1a1a19',
    page: '#0d0d0d',
    textPrimary: '#ffffff',
    textSecondary: '#c3c2b7',
    gridline: '#2c2c2a',
  },
}

export function usePrefersDark() {
  const [dark, setDark] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches,
  )
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = (e) => setDark(e.matches)
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [])
  return dark
}

export function usePalette() {
  const dark = usePrefersDark()
  return dark ? PALETTE.dark : PALETTE.light
}
