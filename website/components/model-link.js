import { createTextE } from './text.js'

export function createModelLinkE(modelInformation, allowLineBreaks=true) {
    const name = modelInformation.short_name ?? modelInformation.model_name

    if (modelInformation.url === undefined) {
        const textE = createTextE(name)
        textE.classList.add('nowrap')
        return textE
    }

    const linkE = document.createElement('a')

    if (allowLineBreaks) {
        linkE.textContent = name.replaceAll('/', '/\u200b').replaceAll('_', '_\u200b')
    } else {
        linkE.textContent = name
        linkE.classList.add('nowrap')
    }

    linkE.href = modelInformation.url

    return linkE
}
