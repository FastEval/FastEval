export function createTableScoreCell(rowE, contentE) {
    const cellE = rowE.insertCell()
    cellE.classList.add('score')
    cellE.appendChild(contentE)
}
