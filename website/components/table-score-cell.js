export function createTableScoreCell(rowE, contentE, relativeScore) {
    const cellE = rowE.insertCell()
    cellE.classList.add('score')

    if (relativeScore === -1) {
        cellE.style['background-color'] = 'black'
    } else if (relativeScore !== undefined && relativeScore !== null) {
        const colorRed = 1 - relativeScore
        const colorGreen = relativeScore
        const color = 'rgb(' + (128 + colorRed * 128) + ',' + (128 + colorGreen * 128) + ',128)'
        cellE.style['background-color'] = color
    }

    cellE.appendChild(contentE)
    return cellE
}
