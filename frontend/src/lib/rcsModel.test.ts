import { describe, expect, it } from 'vitest'
import { dispersionNote, effectMeta, formatNestedTest, isCountModel, modelTitle } from './rcsModel'

describe('rcsModel', () => {
  it('maps each outcome type to its effect measure', () => {
    expect(effectMeta('cox').abbr).toBe('HR')
    expect(effectMeta('logistic').abbr).toBe('OR')
    expect(effectMeta('linear')).toMatchObject({ abbr: 'Δ', refValue: 0, axisType: 'linear' })
    expect(effectMeta('poisson')).toMatchObject({ abbr: 'IRR', refValue: 1, axisType: 'log' })
    expect(effectMeta('quasipoisson').abbr).toBe('IRR')
  })

  it('names the model', () => {
    expect(modelTitle('quasipoisson')).toBe('Quasi-Poisson RCS')
    expect(modelTitle('unknown')).toBe('Logistic RCS')
    expect(isCountModel('poisson') && isCountModel('quasipoisson') && !isCountModel('cox')).toBe(true)
  })

  it('warns about overdispersion only under Poisson', () => {
    expect(dispersionNote('poisson', 2.18)?.warn).toBe(true)
    expect(dispersionNote('poisson', 1.1)).toEqual({ warn: false, text: 'dispersion = 1.10: consistent with Poisson' })
    expect(dispersionNote('quasipoisson', 2.18)?.warn).toBe(false)
    expect(dispersionNote('logistic', 2)).toBeNull()
    expect(dispersionNote('poisson', null)).toBeNull()
  })

  it('formats LR and F nested tests', () => {
    expect(formatNestedTest({ test: 'chi2', lr_stat: 3.621, df: 3 })).toBe('χ²(3) = 3.62')
    expect(formatNestedTest({ lr_stat: 1, df: 2 })).toBe('χ²(2) = 1.00')
    expect(formatNestedTest({ test: 'F', f: 0.549, df: 3, df_resid: 290 })).toBe('F(3, 290) = 0.55')
  })
})
