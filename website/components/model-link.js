import { createTextE } from './text.js'

export function createModelLinkE(modelName, url) {
    if (url === undefined)
        return createTextE(modelName)

    const linkE = document.createElement('a')
    linkE.textContent = modelName
    linkE.href = url
    return linkE
}
