import { createExplanationTextE } from './text.js'

export function createSelectV(label, textContents, values) {
    const selectV = document.createElement('div')
    selectV.appendChild(createExplanationTextE(label + ': '))
    const selectE = document.createElement('select')
    selectV.appendChild(selectE)
    for (let i = 0; i < textContents.length; i++) {
        const optionE = document.createElement('option')
        optionE.value = values[i]
        optionE.textContent = textContents[i]
        selectE.appendChild(optionE)
    }

    return { view: selectV, element: selectE }
}
