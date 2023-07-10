import { parseHash } from '../utils.js'

export function createBackToMainPageE(text='← Back to main page', params={}) {
    const linkE = document.createElement('a')
    linkE.classList.add('back-to-main-page')
    linkE.textContent = text

    const oldItems = parseHash()
    const newItems = new Map()
    if (oldItems.has('branch'))
        newItems.set('branch', oldItems.get('branch'))
    for (const [k, v] of Object.entries(params))
        newItems.set(k, v)
    const href = '#?' + [...newItems.entries()].map(([k, v]) => k + '=' + v).join('&')
    linkE.href = href

    return linkE
}
