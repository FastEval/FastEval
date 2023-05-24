import { computeUpdatedHash } from '../utils.js'

export function createLinkE(text, hashParametersToUpdate) {
    const linkE = document.createElement('a')
    linkE.textContent = text
    linkE.href = '#' + computeUpdatedHash(hashParametersToUpdate)
    return linkE
}
