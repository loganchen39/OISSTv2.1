program read_sic
    implicit none

  ! unsigned(kind=1), dimension(1440, 720, 365) :: sic_ch

  ! character  :: sic_ch(1440, 720, 365)
    character, allocatable  :: sic_ch(:, :, :)
    integer(kind=2)    :: sic(1440, 720, 365)

    ! local vars
    integer :: i, j, k
    character(len=256) :: fn =  &
        "/glade/scratch/lgchen/data/SeaIceConcentration_NSIDC/G02202_V4/NSIDC2OI/interpByCDO/1979_sic_nsidc2oi_v0.2_ok.bin"

    open(unit=10, file=fn, form="unformatted", access="direct", recl=1440*720*365, status="old")
    allocate(sic_ch(1440, 720, 365))
    read(10, rec=1) sic_ch
    do i=1, 365; do j=1, 720; do k=1, 1440
        sic(k, j, i) = ichar(sic_ch(k, j, i))
    end do; end do; end do

    write(*, *) sic(:, 680, 1)

    return
end 



