export function createTableScoreCell(rowE, contentE, color) {
    const cellE = rowE.insertCell()
    cellE.style['background-color'] = color
    cellE.classList.add('score')
    cellE.appendChild(contentE)
}
