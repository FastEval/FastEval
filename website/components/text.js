export function createTextE(...textWithElements) {
    const textE = document.createElement('span')
    for (const textOrElement of textWithElements) {
        if (typeof textOrElement === 'string' || typeof textOrElement === 'number') {
            const textPartE = document.createElement('span')
            textPartE.textContent = textOrElement
            textE.appendChild(textPartE)
        } else {
            textE.appendChild(textOrElement)
        }
    }
    return textE
}
