program read_sic
    implicit none

    integer(kind=1) :: mask(1440, 720)

    ! local vars
    integer :: i, j, k
    character(len=256) :: fn =  &
        "/glade/scratch/lgchen/data/SeaIceConcentration_NSIDC/mask_fromJim/fromDT/mask_oi.bin"
    open(unit=10, file=fn, form="unformatted", access="direct", recl=1440*720, status="old")
    read(10, rec=1) mask
    write(*, *) mask(:, 700)

    return
end 



