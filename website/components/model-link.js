import { createTextE } from './text.js'
import { allowCharacterLineBreaks } from '../utils.js'

export function createModelLinkE(modelInformation, allowLineBreaks=true) {
    const name = modelInformation.short_name ?? modelInformation.model_name

    if (modelInformation.url === undefined) {
        const textE = createTextE(name)
        textE.classList.add('nowrap')
        return textE
    }

    const linkE = document.createElement('a')
    if (allowLineBreaks) {
        linkE.textContent = allowCharacterLineBreaks(name)
    } else {
        linkE.textContent = name
        linkE.classList.add('nowrap')
    }
    linkE.href = modelInformation.url
    return linkE
}
