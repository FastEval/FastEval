import { createTextE } from './text.js'
import { allowCharacterLineBreaks } from '../utils.js'

export function createModelLinkE(modelInformation) {
    if (modelInformation.url === undefined)
        return createTextE(modelInformation.short_name ?? modelInformation.model_name)

    const linkE = document.createElement('a')
    linkE.textContent = allowCharacterLineBreaks(modelInformation.short_name ?? modelInformation.model_name)
    linkE.href = modelInformation.url
    return linkE
}
